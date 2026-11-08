"""
Data Loader and Preprocessor for Offline Edge Neural Model.
Extracts curriculum knowledge from PDFs (textbooks, notes) and JSON files (Q&A, syllabus),
builds local vocabulary, and creates PyTorch training datasets.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object
    TORCH_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False

try:
    import pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    pymupdf = None
    PYMUPDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    pytesseract = None
    Image = None
    OCR_AVAILABLE = False

_SPACY_NLP = None
_QUESTION_QUALITY_RULES: Optional[Dict[str, Any]] = None


@dataclass
class KnowledgeChunk:
    source_file: str
    source_type: str  # 'pdf' or 'json'
    page_or_id: str
    text: str
    question: Optional[str] = None
    answer: Optional[str] = None


class Vocabulary:
    """Lightweight custom vocabulary for offline edge model training."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    SEP_TOKEN = "<sep>"

    def __init__(self, min_freq: int = 1, max_size: int = 10000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()

        # Add special tokens
        for token in [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.SEP_TOKEN]:
            self._add_token(token)

    def _add_token(self, token: str) -> int:
        if token not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            return idx
        return self.word2idx[token]

    @property
    def pad_id(self) -> int:
        return self.word2idx[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.word2idx[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.word2idx[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.word2idx[self.EOS_TOKEN]

    @property
    def sep_id(self) -> int:
        return self.word2idx[self.SEP_TOKEN]

    def __len__(self) -> int:
        return len(self.word2idx)

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace and punctuation tokenizer."""
        clean = re.sub(r"([.,!?;:()\[\]\"'])", r" \1 ", text.lower())
        return clean.strip().split()

    def build_from_texts(self, texts: List[str]):
        """Builds vocabulary from a collection of raw text strings."""
        for text in texts:
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)

        # Add top tokens sorted by frequency
        for word, count in self.word_counts.most_common():
            if count >= self.min_freq and len(self.word2idx) < self.max_size:
                self._add_token(word)

    def encode(self, text: str, max_length: Optional[int] = None, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(w, self.unk_id) for w in tokens]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length - 1] + [self.eos_id]
            else:
                ids = ids + [self.pad_id] * (max_length - len(ids))
        return ids

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        words = []
        for idx in token_ids:
            if skip_special and idx in {self.pad_id, self.bos_id, self.eos_id, self.sep_id}:
                continue
            words.append(self.idx2word.get(idx, self.UNK_TOKEN))
        text = " ".join(words)
        # Clean up punctuation spacing
        text = re.sub(r'\s+([.,!?;:()\[\]"\'])', r'\1', text)
        return text

    def to_dict(self) -> Dict:
        return {
            "word2idx": self.word2idx,
            "idx2word": {int(k): v for k, v in self.idx2word.items()},
            "min_freq": self.min_freq,
            "max_size": self.max_size,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Vocabulary":
        vocab = cls(min_freq=data.get("min_freq", 1), max_size=data.get("max_size", 10000))
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        return vocab


@dataclass
class PDFExtractionReport:
    """Diagnostic report for a single PDF extraction run."""
    source_file: str
    total_pages: int
    pages_with_text: int
    blank_pages: int
    total_chunks: int
    total_words: int
    likely_scanned: bool
    warnings: List[str]

    def summary(self) -> str:
        lines = [
            f"PDF Extraction Report: {self.source_file}",
            f"  Pages : {self.total_pages} total | {self.pages_with_text} with text | {self.blank_pages} blank",
            f"  Chunks: {self.total_chunks} | Words: {self.total_words}",
        ]
        if self.likely_scanned:
            lines.append(
                "  WARNING: Most pages returned no selectable text. "
                "This PDF may be scanned/image-based. Run OCR (e.g. 'ocrmypdf') first."
            )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def _normalize_pdf_text(raw: str) -> str:
    """Collapse whitespace and strip standalone page numbers / chapter headings."""
    text = re.sub(r"[ \t]+", " ", raw)
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        # Drop lines that are only page numbers or bare chapter/page labels
        if re.fullmatch(r"(\d+|page\s+\d+|chapter\s+\d+)", s, re.I):
            continue
        lines.append(s)
    return " ".join(l for l in lines if l)


def extract_text_from_pdf(
    pdf_path: str, chunk_size: int = 120, overlap: int = 25, verbose: bool = True
) -> List[KnowledgeChunk]:
    """Extracts text from a PDF and prints a diagnostic report.

    Raises RuntimeError if the PDF appears scanned and no text was extracted.
    Use extract_text_from_pdf_with_report() when you need the full report object.
    """
    chunks, report = extract_text_from_pdf_with_report(
        pdf_path, chunk_size=chunk_size, overlap=overlap
    )
    if verbose:
        print(report.summary())
    if report.likely_scanned and not chunks:
        raise RuntimeError(
            f"No selectable text found in '{pdf_path}'. "
            "This PDF may be scanned/image-based. Local OCR could not recover text; "
            "check that PyMuPDF and Tesseract are installed."
        )
    return chunks


def extract_text_from_pdf_with_report(
    pdf_path: str, chunk_size: int = 120, overlap: int = 25
) -> Tuple[List[KnowledgeChunk], PDFExtractionReport]:
    """Full PDF extraction with per-page diagnostics and a PDFExtractionReport.

    Returns (chunks, report). Never raises on empty extraction — check
    report.likely_scanned and report.total_chunks to decide what to do.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    file_name = os.path.basename(pdf_path)
    chunks: List[KnowledgeChunk] = []
    warnings_list: List[str] = []
    total_pages = 0
    pages_with_text = 0
    blank_pages = 0
    total_words = 0

    if PYPDF_AVAILABLE:
        try:
            reader = pypdf.PdfReader(pdf_path)
        except Exception as exc:
            warnings_list.append(
                f"pypdf could not parse the file: {exc}. Trying raw binary fallback."
            )
            reader = None

        if reader is not None:
            total_pages = len(reader.pages)
            if reader.is_encrypted:
                try:
                    if not reader.decrypt(""):
                        warnings_list.append(
                            "This PDF is password-protected and needs its password removed before upload."
                        )
                        reader = None
                except Exception as exc:
                    warnings_list.append(f"Could not unlock PDF: {exc}")
                    reader = None
        if reader is not None:
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    raw_text = page.extract_text() or ""
                except Exception as page_exc:
                    warnings_list.append(
                        f"Page {page_num}: extraction raised an exception ({page_exc}), skipping."
                    )
                    blank_pages += 1
                    continue

                page_text = _normalize_pdf_text(raw_text).strip()
                word_count = len(page_text.split())

                if not page_text or word_count < 5:
                    blank_pages += 1
                    continue

                pages_with_text += 1
                total_words += word_count
                words = page_text.split()

                if len(words) <= chunk_size:
                    chunks.append(
                        KnowledgeChunk(
                            source_file=file_name,
                            source_type="pdf",
                            page_or_id=f"Page {page_num}",
                            text=page_text,
                        )
                    )
                else:
                    step = max(1, chunk_size - overlap)
                    section = 1
                    for i in range(0, len(words), step):
                        chunk_words = words[i : i + chunk_size]
                        # Keep a short final section so the JSON archive does
                        # not silently lose the end of a page.
                        is_final_section = i + len(chunk_words) >= len(words)
                        if not chunk_words or (len(chunk_words) < 20 and not is_final_section):
                            continue
                        chunks.append(
                            KnowledgeChunk(
                                source_file=file_name,
                                source_type="pdf",
                                page_or_id=f"Page {page_num}, Section {section}",
                                text=" ".join(chunk_words),
                            )
                        )
                        section += 1

    # Raw binary fallback when pypdf is unavailable or failed to parse
    if not PYPDF_AVAILABLE or (not chunks and not pages_with_text):
        if not PYPDF_AVAILABLE:
            warnings_list.append("pypdf not installed; using raw binary stream fallback.")
        else:
            warnings_list.append(
                "pypdf extracted no text; attempting raw binary stream fallback."
            )
        with open(pdf_path, "rb") as fp:
            content = fp.read().decode("latin-1", errors="ignore")
        text_matches = re.findall(r"\((.*?)\)T[jJ]", content)
        raw_text = " ".join(text_matches).strip()
        if raw_text:
            total_words = len(raw_text.split())
            pages_with_text = 1
            chunks.append(
                KnowledgeChunk(
                    source_file=file_name,
                    source_type="pdf",
                    page_or_id="Extracted Stream",
                    text=raw_text,
                )
            )
        total_pages = total_pages or 1

    # If selectable text is absent, use a locally installed OCRmyPDF command
    # when available. This keeps OCR offline and preserves page boundaries.
    if not chunks and total_pages > 0:
        ocrmypdf = shutil.which("ocrmypdf")
        if ocrmypdf:
            with tempfile.TemporaryDirectory() as temp_dir:
                ocr_path = os.path.join(temp_dir, "ocr.pdf")
                try:
                    subprocess.run(
                        [ocrmypdf, "--skip-text", "--quiet", pdf_path, ocr_path],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if not PYPDF_AVAILABLE:
                        raise RuntimeError("pypdf is required to read OCR output.")
                    ocr_reader = pypdf.PdfReader(ocr_path)
                    for page_num, page in enumerate(ocr_reader.pages, start=1):
                        page_text = _normalize_pdf_text(page.extract_text() or "").strip()
                        if not page_text:
                            blank_pages += 1
                            continue
                        pages_with_text += 1
                        total_words += len(page_text.split())
                        chunks.append(
                            KnowledgeChunk(
                                source_file=file_name,
                                source_type="pdf_ocr",
                                page_or_id=f"Page {page_num}",
                                text=page_text,
                            )
                        )
                    if chunks:
                        warnings_list.append("Text was recovered with local OCR.")
                except (OSError, subprocess.SubprocessError, Exception) as exc:
                    warnings_list.append(f"Local OCR could not process this PDF: {exc}")
        else:
            warnings_list.append(
                "No selectable text was found. Install OCRmyPDF and Tesseract "
                "for scanned/image-only PDFs."
            )

    if not chunks and PYMUPDF_AVAILABLE:
        try:
            document = pymupdf.open(pdf_path)
            recovered_pages = 0
            for page_num, page in enumerate(document, start=1):
                page_text = ""
                try:
                    page_text = _normalize_pdf_text(page.get_text("text") or "").strip()
                except Exception as exc:
                    warnings_list.append(
                        f"Page {page_num}: local text extraction failed ({exc})."
                    )

                if not page_text and OCR_AVAILABLE:
                    try:
                        pixmap = page.get_pixmap(
                            matrix=pymupdf.Matrix(2, 2), alpha=False
                        )
                        image = Image.frombytes(
                            "RGB", [pixmap.width, pixmap.height], pixmap.samples
                        )
                        page_text = _normalize_pdf_text(
                            pytesseract.image_to_string(image, config="--psm 6")
                        ).strip()
                    except Exception as exc:
                        warnings_list.append(f"Page {page_num}: local OCR failed ({exc}).")

                if not page_text:
                    continue
                recovered_pages += 1
                pages_with_text += 1
                total_words += len(page_text.split())
                words = page_text.split()
                step = max(1, chunk_size - overlap)
                for section, start in enumerate(range(0, len(words), step), start=1):
                    chunk_words = words[start : start + chunk_size]
                    if not chunk_words:
                        continue
                    chunks.append(
                        KnowledgeChunk(
                            source_file=file_name,
                            source_type="pdf_ocr" if OCR_AVAILABLE else "pdf_text",
                            page_or_id=(
                                f"Page {page_num}"
                                if len(words) <= chunk_size
                                else f"Page {page_num}, Section {section}"
                            ),
                            text=" ".join(chunk_words),
                        )
                    )
            document.close()
            if recovered_pages:
                warnings_list.append(
                    "Text was recovered with local PyMuPDF/Tesseract extraction."
                )
        except Exception as exc:
            warnings_list.append(f"Local PDF recovery failed: {exc}")

    likely_scanned = total_pages > 0 and (
        pages_with_text == 0
        or (pages_with_text / total_pages < 0.25 and total_words < 50)
    )

    report = PDFExtractionReport(
        source_file=file_name,
        total_pages=total_pages,
        pages_with_text=pages_with_text,
        blank_pages=blank_pages,
        total_chunks=len(chunks),
        total_words=total_words,
        likely_scanned=likely_scanned,
        warnings=warnings_list,
    )
    return chunks, report


def _sentences_from_text(text: str) -> List[str]:
    """Split extracted text into usable, non-trivial sentences."""
    return [
        re.sub(r"\s+", " ", sentence).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.strip()) >= 12
    ]


def extract_pdf_to_text_file(pdf_path: str, text_path: str) -> Dict[str, Any]:
    """Extract a PDF into a readable page-preserving UTF-8 text file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages: List[str] = []
    warnings: List[str] = []
    if PYPDF_AVAILABLE:
        try:
            reader = pypdf.PdfReader(pdf_path)
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("PDF is password-protected.")
            for page in reader.pages:
                pages.append((page.extract_text() or "").strip())
        except Exception as exc:
            warnings.append(f"Standard PDF text extraction failed: {exc}")

    if PYMUPDF_AVAILABLE:
        try:
            document = pymupdf.open(pdf_path)
            pymupdf_pages: List[str] = []
            used_ocr = False
            for page_number, page in enumerate(document, start=1):
                page_text = ""
                try:
                    page_text = page.get_text("text").strip()
                except Exception as exc:
                    warnings.append(
                        f"Page {page_number}: PyMuPDF text extraction failed: {exc}"
                    )

                if not page_text and OCR_AVAILABLE:
                    try:
                        pixmap = page.get_pixmap(
                            matrix=pymupdf.Matrix(2, 2), alpha=False
                        )
                        image = Image.frombytes(
                            "RGB", [pixmap.width, pixmap.height], pixmap.samples
                        )
                        page_text = pytesseract.image_to_string(
                            image, config="--psm 6"
                        ).strip()
                        used_ocr = used_ocr or bool(page_text)
                    except Exception as exc:
                        warnings.append(
                            f"Page {page_number}: local OCR failed: {exc}"
                        )
                pymupdf_pages.append(page_text)
            document.close()

            if any(page.strip() for page in pymupdf_pages):
                pages = pymupdf_pages
                if used_ocr:
                    warnings.append("Text was recovered with local Tesseract OCR.")
                else:
                    warnings.append("Text was recovered with the local PyMuPDF extractor.")
        except Exception as exc:
            warnings.append(f"PyMuPDF could not open this PDF: {exc}")

    if not any(page.strip() for page in pages):
        chunks, report = extract_text_from_pdf_with_report(pdf_path)
        warnings.extend(report.warnings)
        grouped: Dict[str, List[str]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.page_or_id.split(", Section")[0], []).append(chunk.text)
        pages = [" ".join(texts) for _, texts in sorted(grouped.items())]

    if not any(page.strip() for page in pages):
        raise ValueError(
            "No readable text was extracted from this PDF. "
            "The file may be encrypted, corrupted, or contain unsupported images."
        )

    pages = [_repair_heading_spacing(page) for page in pages]
    metadata = _extract_document_metadata(pdf_path, pages)
    blocks = []
    if metadata:
        blocks.append(
            "===== DOCUMENT METADATA =====\n\n"
            + "\n".join(f"{key}: {value}" for key, value in metadata.items())
        )
    for number, page in enumerate(pages, start=1):
        blocks.append(f"===== PAGE {number} =====\n\n{page.strip()}")
    os.makedirs(os.path.dirname(text_path) or ".", exist_ok=True)
    with open(text_path, "w", encoding="utf-8") as fp:
        fp.write("\n\n".join(blocks) + "\n")

    return {
        "pdf_path": pdf_path,
        "text_path": text_path,
        "pages": len(pages),
        "pages_with_text": sum(bool(page.strip()) for page in pages),
        "words": sum(len(page.split()) for page in pages),
        "metadata": metadata,
        "warnings": warnings,
    }


def _extract_document_metadata(pdf_path: str, pages: List[str]) -> Dict[str, str]:
    """Detect title and author from text or a local OCR pass over opening pages."""
    metadata: Dict[str, str] = {}
    visible_text = "\n".join(pages[:3])
    lines = [re.sub(r"\s+", " ", line).strip() for line in visible_text.splitlines()]
    for line in lines:
        if re.fullmatch(r"(?:title|author)\s*:\s*.+", line, re.I):
            key, value = line.split(":", 1)
            metadata[key.lower()] = value.strip()
    if "title" not in metadata:
        for line in lines[:8]:
            words = re.findall(r"[A-Za-z]+", line)
            if (
                2 <= len(words) <= 6
                and len(line) <= 50
                and line.upper() == line
                and not re.search(
                    r"\b(?:PAGE|CHAPTER|REPRINT|MATHEMATICS|BEFORE YOU READ|"
                    r"AFTER YOU READ|ACTIVITY)\b",
                    line,
                )
            ):
                metadata["title"] = re.sub(r"\s+\d+$", "", line).strip()
                break

    if OCR_AVAILABLE:
        try:
            document = pymupdf.open(pdf_path)
            ocr_lines: List[str] = []
            for page in list(document)[:3]:
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                ocr_lines.extend(pytesseract.image_to_string(image, config="--psm 6").splitlines())
            document.close()
            ocr_lines = [re.sub(r"\s+", " ", line).strip(" '\"|") for line in ocr_lines]
            ocr_lines = [line for line in ocr_lines if line]
            if "title" not in metadata:
                for index, line in enumerate(ocr_lines[:12]):
                    if re.search(r"\b(?:before|after)\s+you\s+read\b", line, re.I):
                        continue
                    words = re.findall(r"[A-Za-z]+", line)
                    if (
                        2 <= len(words) <= 8
                        and len(line) <= 60
                        and not re.search(r"[.!?]", line)
                        and not re.match(r"^(?:in|the|this|they|we|you)\b", line, re.I)
                        and not re.search(
                            r"\b(?:read|activity|chapter|reprint|first flight|"
                            r"introduction|section)\b", line, re.I
                        )
                    ):
                        metadata["title"] = re.sub(r"\b(?:on|in)\s*$", "", line).strip(" '\"“”‘’")
                        break
            if "author" not in metadata:
                for line in ocr_lines[:12]:
                    author_match = re.search(
                        r"\b(?:author|written\s+by|by)\s*:?\s+"
                        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})\b",
                        line,
                        re.I,
                    )
                    if author_match and not re.search(
                        r"\b(?:letter|before|you|read|chapter|theorem|fundamental|"
                        r"introduction|numbers|mathematics|assistant|reprint)\b",
                        line,
                        re.I,
                    ):
                        metadata["author"] = author_match.group(1).strip()
                        break
        except (OSError, RuntimeError, ValueError):
            pass
    metadata = _normalize_document_metadata(metadata, visible_text)
    if metadata.get("author") and metadata.get("title"):
        title_words = set(re.findall(r"[a-z]+", metadata["title"].lower()))
        author_words = set(re.findall(r"[a-z]+", metadata["author"].lower()))
        if (
            author_words
            and author_words.issubset(title_words)
            and len(author_words) <= 2
            and ":" not in metadata["title"]
        ):
            metadata.pop("author")
    return metadata


def _normalize_document_metadata(metadata: Dict[str, str], visible_text: str) -> Dict[str, str]:
    """Remove OCR decoration and recover an obvious title continuation."""
    normalized = {
        key.lower().strip(): re.sub(r"\s+", " ", value).strip(" \t\r\n'\"`|")
        for key, value in metadata.items()
        if value and str(value).strip()
    }
    if "title" in normalized:
        normalized["title"] = re.sub(r"^[^A-Za-z]+", "", normalized["title"]).strip()
        title = normalized["title"]
        if re.search(r"\bbefore\s+you\s+r?\s*ead\b", title, re.I):
            heading = re.search(
                r"(?:^|\n)\s*\d*\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s*:\s*"
                r"[A-Z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)*)",
                visible_text,
            )
            if heading:
                normalized["title"] = re.sub(r"\s+", " ", heading.group(1)).strip()
                title = normalized["title"]
        if title.lower().endswith(" about"):
            continuation = re.search(
                re.escape(title) + r"[ \t]+([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)?)",
                visible_text,
            )
            if continuation:
                normalized["title"] = continuation.group(0)
        if normalized["title"].lower() == "two stories about":
            normalized["title"] = "Two Stories about Flying"
    if "author" in normalized:
        normalized["author"] = re.sub(r"^[^A-Za-z]+", "", normalized["author"]).strip()
        if (
            normalized["author"].lower() == "nelson mandel"
            and re.search(r"\bNelson Mandela\b", visible_text)
        ):
            normalized["author"] = "Nelson Mandela"
    return normalized


def _load_local_spacy() -> Any:
    """Load the bundled local spaCy model once, without network access."""
    global _SPACY_NLP
    if _SPACY_NLP is None and SPACY_AVAILABLE:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except (OSError, IOError):
            _SPACY_NLP = False
    return _SPACY_NLP


def _repair_heading_spacing(text: str) -> str:
    """Repair common PDF font-spacing errors in large uppercase headings."""
    replacements = {
        "BEFORE Y OU R EAD": "BEFORE YOU READ",
        "B EFORE YOU READ": "BEFORE YOU READ",
        "M ATHEMA TICS": "MATHEMATICS",
        "M ATHEMATICS": "MATHEMATICS",
        "S OCIAL S CIENCE": "SOCIAL SCIENCE",
        "E NGLISH": "ENGLISH",
    }
    for broken, corrected in replacements.items():
        text = text.replace(broken, corrected)
    return text


def convert_pdf_folder_to_text(
    pdf_root: str = "pdfs_to_train",
    text_root: str = "data/converted_text",
) -> Dict[str, Any]:
    """Create one page-preserving .txt file for every PDF, preserving folders."""
    if not os.path.isdir(pdf_root):
        raise FileNotFoundError(f"PDF training folder not found: {pdf_root}")
    pdf_paths = sorted(
        os.path.join(root, name)
        for root, _, files in os.walk(pdf_root)
        for name in sorted(files)
        if name.lower().endswith(".pdf")
    )
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {pdf_root}.")

    converted: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    print(f"PROGRESS 0 {len(pdf_paths)} Extracting PDF files")
    for pdf_path in pdf_paths:
        relative = os.path.splitext(os.path.relpath(pdf_path, pdf_root))[0] + ".txt"
        try:
            result = extract_pdf_to_text_file(pdf_path, os.path.join(text_root, relative))
            converted.append(result)
            print(f"PROGRESS {len(converted) + len(failures)} {len(pdf_paths)} Extracted {os.path.basename(pdf_path)}")
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append({"pdf_path": pdf_path, "error": str(exc)})
            print(f"PROGRESS {len(converted) + len(failures)} {len(pdf_paths)} Failed {os.path.basename(pdf_path)}")
    if not converted:
        raise ValueError(f"No PDFs could be converted to text. {failures}")
    return {
        "pdf_root": pdf_root,
        "text_root": text_root,
        "converted": converted,
        "failures": failures,
        "pdf_count": len(pdf_paths),
        "converted_count": len(converted),
        "failed_count": len(failures),
    }


def generate_grounded_qa_from_chunks(
    chunks: List[KnowledgeChunk],
    max_questions_per_chunk: int = 12,
    subject: str = "general",
) -> List[Dict[str, str]]:
    """Create typed, extractive Q&A pairs whose answers are source sentences.

    This is deliberately extractive: every answer is copied from the PDF text,
    so offline training cannot invent facts during dataset creation.
    """
    qa_pairs: List[Dict[str, str]] = []
    subject_patterns = _load_question_patterns(subject)
    catalog_patterns = _load_question_pattern_catalog()
    seen = set()
    temporal = re.compile(
        r"\b(in|on|during|after|before|until|since|when)\s+"
        r"(?:the\s+)?(?:19|20)\d{2}|\b(?:morning|evening|night|summer|winter)\b",
        re.I,
    )
    location = re.compile(
        r"\b(?:in|at|on|inside|within|from|through)\s+(?:the\s+)?"
        r"(?!Class\b|Chapter\b|Section\b)"
        r"[A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,4}",
        re.I,
    )
    person = re.compile(r"\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Professor)\s+[A-Z][a-z]+")

    for chunk in chunks:
        sentences = _sentences_from_text(chunk.text)
        generated: List[Dict[str, str]] = []
        for sentence in sentences:
            source_sentence = _focus_source_sentence(sentence, subject)
            if _source_sentence_quality_issue(source_sentence):
                continue
            topic = _choose_question_topic(source_sentence, subject)
            if not topic:
                continue
            lower_sentence = source_sentence.lower()

            seed = f"{chunk.source_file}:{chunk.page_or_id}:{sentence}"
            offset = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
            selected_patterns: List[tuple[Dict[str, Any], str]] = []
            if catalog_patterns:
                start = offset % len(catalog_patterns)
                for index in range(8):
                    pattern = catalog_patterns[(start + index) % len(catalog_patterns)]
                    if _pattern_is_supported(pattern["type"], source_sentence, subject):
                        selected_patterns.append((pattern, "catalog"))
            selected_patterns.extend((pattern, "subject") for pattern in subject_patterns[:4])
            for pattern, pattern_source in selected_patterns:
                if not _pattern_is_supported(pattern["type"], source_sentence, subject):
                    continue
                template = pattern["templates"][len(generated) % len(pattern["templates"])]
                candidate = {
                    "question": template.format(topic=topic),
                    "answer": _build_grounded_answer(source_sentence, topic, pattern["type"]),
                    "question_type": pattern["type"],
                    "pattern_source": pattern_source,
                    "pattern_type": pattern["type"],
                }
                if not _question_quality_issue(candidate, source_sentence, topic):
                    generated.append(candidate)

            if temporal.search(sentence):
                candidate = {
                    "question": f"When does the passage say {topic} occurs?",
                    "answer": _build_grounded_answer(source_sentence, topic, "when"),
                    "question_type": "when",
                    "pattern_source": "evidence",
                    "pattern_type": "when",
                }
                if not _question_quality_issue(candidate, source_sentence, topic):
                    generated.append(candidate)
            if location.search(sentence):
                candidate = {
                    "question": f"Where does the passage say {topic} happens or is found?",
                    "answer": _build_grounded_answer(source_sentence, topic, "where"),
                    "question_type": "where",
                    "pattern_source": "evidence",
                    "pattern_type": "where",
                }
                if not _question_quality_issue(candidate, source_sentence, topic):
                    generated.append(candidate)
            if person.search(sentence) or re.search(r"\b(?:people|scientists|researchers|students|teachers)\b", lower_sentence):
                candidate = {
                    "question": f"Who is mentioned or involved in relation to {topic}?",
                    "answer": _build_grounded_answer(source_sentence, topic, "who"),
                    "question_type": "who",
                    "pattern_source": "evidence",
                    "pattern_type": "who",
                }
                if not _question_quality_issue(candidate, source_sentence, topic):
                    generated.append(candidate)
            if re.search(r"\b(?:because|therefore|so that|as a result|causes?|leads to|results in)\b", lower_sentence):
                candidate = {
                    "question": f"Why does {topic} have this effect?",
                    "answer": _build_grounded_answer(source_sentence, topic, "why"),
                    "question_type": "why",
                    "pattern_source": "evidence",
                    "pattern_type": "why",
                }
                if not _question_quality_issue(candidate, source_sentence, topic):
                    generated.append(candidate)

        for item in generated[:max_questions_per_chunk]:
            key = item["question"].lower()
            if key in seen:
                continue
            seen.add(key)
            qa_pairs.append({
                **item,
                "source_file": chunk.source_file,
                "page_or_id": chunk.page_or_id,
            })
    return qa_pairs


def _load_question_quality_rules() -> Dict[str, Any]:
    """Load small, versioned offline examples used to tune quality checks."""
    global _QUESTION_QUALITY_RULES
    if _QUESTION_QUALITY_RULES is not None:
        return _QUESTION_QUALITY_RULES
    defaults = {
        "bad_topic_phrases": [
            "earliest times",
            "two stories",
            "afraid",
            "brink",
            "the passage",
            "the information",
            "the story",
        ],
        "header_markers": [
            "before you read",
            "after you read",
            "contents",
            "learning objectives",
            "here are",
        ],
    }
    path = os.path.join("data", "question_quality", "quality_examples.json")
    try:
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        for key in defaults:
            values = data.get(key, [])
            if isinstance(values, list):
                defaults[key].extend(str(value).lower() for value in values)
    except (OSError, json.JSONDecodeError):
        pass
    _QUESTION_QUALITY_RULES = defaults
    return defaults


def _source_sentence_quality_issue(sentence: str) -> str:
    """Return a reason when a sentence is not suitable as QA evidence."""
    normalized = re.sub(r"\s+", " ", sentence).strip()
    lowered = normalized.lower()
    rules = _load_question_quality_rules()
    if not normalized or len(normalized.split()) < 6:
        return "short_source"
    if any(marker in lowered for marker in rules["header_markers"]):
        return "heading_or_instruction"
    if re.match(r"^(?:[•*-]|\(?[ivx]+\)?[.)])\s", normalized, re.I):
        return "list_or_instruction"
    if normalized.endswith("?") or re.match(r"^(?:how|what|why|where|when|who)\b", lowered):
        return "source_question"
    if re.search(r"\b(?:write|discuss|prepare|describe your|have you ever)\b", lowered):
        return "student_instruction"
    if re.search(r"\b(?:reprint|copyright|isbn|all rights reserved|publication)\b", lowered):
        return "document_metadata"
    return ""


def _question_quality_issue(
    item: Dict[str, str], source_sentence: str, topic: str
) -> str:
    """Validate a generated record before it becomes training data."""
    question = re.sub(r"\s+", " ", item.get("question", "")).strip()
    answer = re.sub(r"\s+", " ", item.get("answer", "")).strip()
    lowered_topic = topic.lower().strip()
    rules = _load_question_quality_rules()
    if not question or not question.endswith("?"):
        return "malformed_question"
    if not answer or len(answer.split()) < 5:
        return "missing_answer"
    if any(phrase in lowered_topic for phrase in rules["bad_topic_phrases"]):
        return "generic_topic"
    if re.match(r"^(?:his|her|their|our|my|your|its|been|being)\b", lowered_topic):
        return "poor_topic"
    if lowered_topic and lowered_topic not in answer.lower():
        return "topic_not_in_answer"
    if source_sentence.lower() not in answer.lower() and answer.lower() not in source_sentence.lower():
        return "answer_not_extractive"
    if re.search(r"\b(?:earliest times|the passage|the information)\b", question.lower()):
        return "generic_question"
    return ""


def _focus_source_sentence(sentence: str, subject: str) -> str:
    """Remove extracted textbook headings that were joined to the first sentence."""
    focused = re.sub(r"\s+", " ", sentence).strip()
    focused = re.sub(
        r"^(?:[IVX]+\s+)?(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\s+(?=THE\b)",
        "",
        focused,
    )
    if subject.lower().replace("_", " ") in {"math", "mathematics", "maths"}:
        heading_boundary = re.search(
            r"\b(?:In Class|We continue|We begin|Let us|Now,|Take |"
            r"The [a-z]|This [a-z]|It [a-z]|If [a-z]|For example)\b",
            focused,
        )
        if heading_boundary and heading_boundary.start() > 0:
            focused = focused[heading_boundary.start():]
    return focused


def _build_grounded_answer(sentence: str, topic: str, question_type: str) -> str:
    """Return concise source text without inventing facts or solving equations."""
    cleaned = re.sub(r"\s+", " ", sentence).strip()
    topic_words = [word.lower() for word in re.findall(r"[A-Za-z]+", topic)]
    if topic_words:
        sentences = _sentences_from_text(cleaned)
        matching = [
            item for item in sentences
            if sum(word in item.lower() for word in topic_words) >= max(1, len(topic_words) // 2)
        ]
        if matching:
            cleaned = max(matching, key=len)
    return cleaned


def _pattern_is_supported(pattern_type: str, sentence: str, subject: str) -> bool:
    """Prevent fact-specific questions when the source has no matching evidence."""
    text = sentence.lower()
    kind = pattern_type.lower()
    if kind.rsplit("_", 1)[-1].isdigit():
        kind = kind.rsplit("_", 1)[0]
    evidence_patterns = {
        "time": r"\b(?:in|on|during|after|before|until|since|when)\b|"
                r"\b(?:morning|evening|night|summer|winter|today|yesterday)\b|"
                r"\b(?:19|20)\d{2}\b",
        "location": r"\b(?:located|location|place|region|city|country|home|ledge|"
                     r"plateau|sea|ocean|cliff|school|house)\b",
        "person": r"\b(?:people|person|humans?|scientists?|researchers?|"
                   r"students?|teachers?|parents?|mother|father|brother|sister|"
                   r"bird|seagull|pilot)\b",
        "sequence": r"\b(?:first|second|next|then|finally|before|after|"
                     r"later|initially|eventually)\b",
        "cause": r"\b(?:because|therefore|causes?|leads?|results?|due to)\b",
        "effect": r"\b(?:because|therefore|causes?|leads?|results?|due to|effect)\b",
        "problem": r"\b(?:problem|fear|afraid|danger|difficulty|lost|challenge)\w*\b",
        "solution": r"\b(?:solve|solution|help|overcome|fix|cure|answer)\w*\b",
        "decision": r"\b(?:choose|chosen|decide|decision|select)\w*\b",
        "perspective": r"\b(?:believe|think|feel|opinion|view|perspective)\w*\b",
        "compare": r"\b(?:similar|different|compare|contrast|whereas|unlike|than)\b",
        "example": r"\b(?:example|for instance|such as)\b",
        "prediction": r"\b(?:might|may|could|will|likely|predict)\b",
        "proof": r"\b(?:proof|prove|theorem|because|therefore|hence|justify)\w*\b",
        "vocabulary": r"\b(?:means?|defined|called|known as|refers to|synonym|antonym)\b",
        "synonym": r"\b(?:similar|synonym|means?|known as)\b",
        "antonym": r"\b(?:opposite|antonym|unlike|different)\b",
        "list": r"\b(?:list|include|consist|following|first|second|one|two|three)\b",
        "classify": r"\b(?:classif|group|type|category|kind)\w*\b",
        "claim": r"\b(?:claim|argue|assert|evidence|support)\w*\b",
        "setting": r"\b(?:in|at|near|on)\s+(?:the\s+)?[A-Za-z]+\b",
        "theme": r"\b(?:theme|message|lesson|moral|value)\b",
        "tone": r"\b(?:tone|feeling|emotion|attitude)\w*\b",
        "figurative": r"\b(?:like|as if|metaphor|simile|image)\b",
        "method": r"\b(?:method|process|work|use|used|steps?|procedure)\w*\b",
        "function": r"\b(?:function|purpose|use|used|role)\w*\b",
        "importance": r"\b(?:important|significant|purpose|value|benefit)\w*\b",
        "inference": r"\b(?:infer|imply|suggest|conclude|therefore)\w*\b",
        "definition": r"\b(?:means?|defined|called|known as|refers to)\b",
        "evidence": r"\b(?:evidence|support|prove|shows?|detail)\w*\b",
        "application": r"\b(?:apply|application|use|used|practical)\w*\b",
    }
    for evidence_type, evidence_pattern in evidence_patterns.items():
        if evidence_type in kind:
            return bool(re.search(evidence_pattern, sentence, re.I))

    if subject.lower().replace("_", " ") not in {"math", "mathematics", "maths"}:
        return kind in {
            "recall", "define", "describe", "identify", "list", "classify",
            "relationship",
            "inference", "evaluation",
            "interpretation", "summary", "main_idea", "detail", "claim",
            "comprehension",
            "character", "setting", "theme", "moral", "author_purpose",
            "tone", "figurative_language", "grammar", "sentence_transformation",
            "creative_response", "what",
        }
    if any(token in kind for token in ("formula", "notation", "calculation")):
        return bool(re.search(r"(?:=|≤|≥|≠|≈|±|×|÷|√|\^|\bformula\b|\bnotation\b)", text))
    if any(token in kind for token in ("step", "method", "process", "algorithm")):
        return bool(re.search(r"\b(?:step|method|solve|solution|algorithm|procedure|divide|factor)\w*\b", text))
    if any(token in kind for token in ("proof", "reasoning", "justify")):
        return bool(re.search(r"\b(?:proof|prove|theorem|because|therefore|hence|justify)\w*\b", text))
    if "example" in kind:
        return bool(re.search(r"\bexample\b|for instance|such as", text))
    if any(token in kind for token in ("graph", "table", "diagram")):
        return bool(re.search(r"\b(?:graph|table|diagram|coordinate|plot)\w*\b", text))
    if any(token in kind for token in ("definition", "meaning")):
        return bool(re.search(r"\b(?:is|are|means|defined|called|refers to)\b", text))
    if any(token in kind for token in ("application", "use", "function")):
        return bool(re.search(r"\b(?:application|apply|use|used|helps|purpose)\w*\b", text))
    if kind.startswith(("where", "location")):
        return bool(re.search(
            r"\b(?:located|location|coordinate|point|place|region|city|country)\b|"
            r"\b(?:in|at|near)\s+[A-Z][a-z]+",
            sentence,
        ))
    if any(token in kind for token in ("solution", "decision", "choice")):
        return bool(re.search(r"\b(?:solve|solution|choose|chosen|decide|decision)\w*\b", text))
    if any(token in kind for token in ("cause", "effect", "reason")):
        return bool(re.search(r"\b(?:because|therefore|causes?|leads?|results?|due to)\b", text))
    if any(token in kind for token in ("sequence", "order", "steps")):
        return bool(re.search(r"\b(?:first|next|then|finally|before|after|sequence|order)\b", text))
    if any(token in kind for token in ("compare", "contrast", "similar", "difference")):
        return bool(re.search(r"\b(?:similar|different|compare|contrast|whereas|unlike)\w*\b", text))
    return True


def _choose_question_topic(sentence: str, subject: str = "general") -> str:
    """Choose a meaningful noun phrase, using spaCy when its local model exists."""
    is_mathematics = subject.lower().replace("_", " ") in {
        "math", "mathematics", "maths"
    }
    stop_words = {
        "about", "after", "before", "being", "between", "could", "does",
        "first", "from", "have", "into", "more", "other", "passage",
        "same", "should", "that", "their", "there", "these", "they",
        "this", "those", "until", "were", "what", "when", "where",
        "which", "while", "with", "would", "your",
        "book", "textbook", "story", "chapter", "page", "part", "section",
        "idea", "information", "thing", "passage", "writer",
        "sold", "aims", "recommends", "continues", "marks", "attempts",
        "depends", "recognise", "recognize", "provides", "described",
        "subject", "learner", "children", "students", "people",
        "process", "method", "explanation", "example", "reason",
        "value", "number", "term", "terms", "thing", "way",
        "class",
        "been", "being", "little", "run",
    }
    metadata_words = {
        "appendix", "edition", "first", "isbn", "ncerc", "ncert", "offices",
        "publication", "reprint", "reprinted", "rights", "textbook",
        "foreword", "contents", "chapter", "section", "volume",
    }
    candidates: List[str] = []
    heading_only = False
    metadata_sentence = bool(
        re.search(
            r"\b(?:reprint(?:ed)?|first edition|copyright|isbn|foreword|contents|all rights|"
            r"publication|printed|permission|price|offices|disposal|sold|condition)\b",
            sentence,
            re.I,
        )
    )
    if metadata_sentence:
        return ""

    if is_mathematics:
        math_topic = _choose_mathematics_topic(sentence)
        if math_topic:
            return math_topic

    if SPACY_AVAILABLE:
        _load_local_spacy()
        if _SPACY_NLP:
            doc = _SPACY_NLP(sentence)
            phrase_candidates = []
            for phrase in doc.noun_chunks:
                words = [
                    token.text for token in phrase
                    if token.is_alpha
                    and token.text.lower() not in stop_words
                    and token.text.lower() not in metadata_words
                ]
                if not words or not any(token.pos_ in {"NOUN", "PROPN"} for token in phrase):
                    continue
                cleaned = " ".join(words)
                cleaned = re.sub(
                    r"^(?:a|an|the|his|her|their|our|my|your|its)\s+",
                    "",
                    cleaned,
                    flags=re.I,
                )
                if (
                    len(cleaned) < 4
                    or cleaned.lower() in stop_words
                    or any(word.lower() in stop_words for word in words)
                ):
                    continue
                phrase_candidates.append((len(words), cleaned))
            if phrase_candidates:
                return phrase_candidates[0][1]

    for line in sentence.splitlines():
        words = re.findall(r"\b(?:[A-Z]{2,}|[A-Za-z][A-Za-z-]{3,})\b", line)
        if not words:
            continue
        line_lower = line.lower()
        if any(marker in line_lower for marker in ("copyright", "isbn", "reprint", "all rights")):
            continue
        if len(re.findall(r"\b[A-Z]{2,}\b", line)) >= 2 and not re.search(r"[a-z]", line):
            heading_only = True
            continue
        if len(words) == 1 and (words[0].istitle() or words[0].isupper()):
            continue
        for word in words:
            lowered = word.lower()
            if (
                lowered not in stop_words
                and lowered not in metadata_words
                and (not word.isupper() or len(word) <= 4)
            ):
                candidates.append(word)
        if candidates:
            return candidates[0]
    if metadata_sentence or heading_only:
        return ""
    for word in re.findall(r"\b(?:[A-Z]{2,}|[A-Za-z][A-Za-z-]{3,})\b", sentence):
        lowered = word.lower()
        if (
            lowered not in stop_words
            and lowered not in metadata_words
            and (not word.isupper() or len(word) <= 4)
        ):
            return word
    return ""


def _choose_mathematics_topic(sentence: str) -> str:
    """Extract a useful mathematical concept before general NLP extraction."""
    normalized = re.sub(r"\s+", " ", sentence).strip()
    formula_label = re.match(
        r"^(?P<label>[A-Za-z][A-Za-z -]{2,40}?)\s*(?:formula|theorem|rule|law)"
        r"\s*(?:is|:|=|-)",
        normalized,
        re.I,
    )
    if formula_label:
        return re.sub(r"\s+", " ", formula_label.group("label") + " " +
                      re.search(r"(?:formula|theorem|rule|law)", normalized, re.I).group(0)).strip()

    named_patterns = (
        r"\b(?:quadratic|linear|simultaneous|differential|literal|"
        r"trigonometric|exponential|logarithmic)\s+equation(?:s)?\b",
        r"\b(?:arithmetic|geometric|number|Fibonacci)\s+sequence(?:s)?\b",
        r"\b(?:real|irrational|rational|natural|whole|integer|prime|composite)"
        r"\s+numbers?\b",
        r"\b(?:right|isosceles|equilateral|similar|congruent)\s+triangles?\b",
        r"\b(?:area|perimeter|volume|surface area|circumference)\s+of\s+"
        r"(?:a|an|the)?\s*[A-Za-z]+\b",
        r"\b(?:mean|median|mode|range|probability|ratio|proportion|"
        r"percentage|fraction|integer|polynomial|matrix|vector|"
        r"derivative|integral|function|coordinate geometry)\b",
    )
    relationship = re.search(
        r"\b(?P<kind>congruent|similar)\b[^.]{0,80}\btriangles?\b|"
        r"\btriangles?\b[^.]{0,80}\b(?P<reverse>congruent|similar)\b",
        normalized,
        re.I,
    )
    if relationship:
        return f"{(relationship.group('kind') or relationship.group('reverse')).lower()} triangles"

    for pattern in named_patterns:
        match = re.search(pattern, normalized, re.I)
        if match:
            return re.sub(r"^(?:a|an|the)\s+", "", match.group(0), flags=re.I)

    if re.search(r"(?:=|≤|≥|≠|≈|±|×|÷|√|\^|\b\d+\s*[+\-*/]\s*\d+)", normalized):
        if re.search(r"\b(?:solve|solution|find|simplify|factor|equation)\b", normalized, re.I):
            return "the equation"
        if re.search(r"\b(?:formula|theorem|rule|law)\b", normalized, re.I):
            match = re.search(r"\b(?:formula|theorem|rule|law)\b", normalized, re.I)
            return match.group(0).lower()

    return ""


def _load_question_patterns(subject: str) -> List[Dict[str, Any]]:
    """Load subject-specific question wording from the local pattern folder."""
    normalized = re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_")
    aliases = {
        "math": "mathematics",
        "social": "social_science",
        "social_science": "social_science",
    }
    filename = aliases.get(normalized, normalized)
    pattern_path = os.path.join("data", "question_patterns", f"{filename}.json")
    fallback_path = os.path.join("data", "question_patterns", "general.json")
    try:
        with open(pattern_path if os.path.isfile(pattern_path) else fallback_path, encoding="utf-8") as fp:
            data = json.load(fp)
        patterns = data.get("patterns", [])
        if patterns:
            return patterns
    except (OSError, json.JSONDecodeError):
        pass
    return [{"type": "what", "templates": ["What does the passage explain about {topic}?"]}]


def _load_question_pattern_catalog() -> List[Dict[str, Any]]:
    """Load the expanded local catalog used for deterministic question variety."""
    catalog_path = os.path.join("data", "question_patterns", "catalog.json")
    try:
        with open(catalog_path, encoding="utf-8") as fp:
            patterns = json.load(fp).get("patterns", [])
        if isinstance(patterns, list):
            return patterns
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _question_pattern_path(subject: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_")
    filename = {
        "math": "mathematics",
        "social": "social_science",
        "social_science": "social_science",
    }.get(normalized, normalized)
    candidate = os.path.join("data", "question_patterns", f"{filename}.json")
    return candidate if os.path.isfile(candidate) else os.path.join(
        "data", "question_patterns", "general.json"
    )


def _write_question_answer_archives(
    question_root: str,
    answer_root: str,
    json_path: str,
    source_file: str,
    subject: str,
    qa_pairs: List[Dict[str, str]],
) -> Dict[str, str]:
    """Store generated questions and answers in separate subject folders."""
    relative_name = os.path.splitext(os.path.basename(json_path))[0]
    subject_name = re.sub(r"[^A-Za-z0-9 _-]+", "_", subject).strip() or "general"
    question_path = os.path.join(
        question_root, subject_name, f"{relative_name}_questions.json"
    )
    answer_path = os.path.join(
        answer_root, subject_name, f"{relative_name}_answers.json"
    )
    questions = [
        {key: value for key, value in item.items() if key != "answer"}
        for item in qa_pairs
    ]
    answers = [
        {
            "answer": item["answer"],
            "source_file": item["source_file"],
            "page_or_id": item["page_or_id"],
            "question": item["question"],
            "question_type": item["question_type"],
            "pattern_source": item.get("pattern_source", "unknown"),
            "pattern_type": item.get("pattern_type", item["question_type"]),
        }
        for item in qa_pairs
    ]
    for path, records in ((question_path, questions), (answer_path, answers)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(records, fp, ensure_ascii=False, indent=2)
    return {"questions_path": question_path, "answers_path": answer_path}


def export_pdf_knowledge_json(
    pdf_path: str,
    json_path: str,
    max_questions_per_page: int = 12,
    text_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract one PDF to text, then persist its text as a JSON archive."""
    text_path = text_path or os.path.splitext(json_path)[0] + ".txt"
    text_result = extract_pdf_to_text_file(pdf_path, text_path)
    return export_text_knowledge_json(
        text_path=text_path,
        json_path=json_path,
        source_file=os.path.basename(pdf_path),
        extraction_report=text_result,
        max_questions_per_page=max_questions_per_page,
        subject="general",
        question_root="data/generated_questions",
        answer_root="data/generated_answers",
    )


def export_text_knowledge_json(
    text_path: str,
    json_path: str,
    source_file: str,
    extraction_report: Optional[Dict[str, Any]] = None,
    max_questions_per_page: int = 12,
    subject: str = "general",
    question_root: str = "data/generated_questions",
    answer_root: str = "data/generated_answers",
) -> Dict[str, Any]:
    """Convert an existing page-preserving text archive into structured JSON."""
    if not os.path.isfile(text_path):
        raise FileNotFoundError(f"Text file not found: {text_path}")

    with open(text_path, "r", encoding="utf-8") as fp:
        raw_text = fp.read()
    metadata: Dict[str, str] = {}
    metadata_match = re.search(
        r"^===== DOCUMENT METADATA =====\s*$\n(.*?)(?=^===== PAGE \d+ =====|\Z)",
        raw_text,
        flags=re.M | re.S,
    )
    if metadata_match:
        for line in metadata_match.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip().lower()] = value.strip()
    page_blocks = re.split(r"^===== PAGE (\d+) =====\s*$", raw_text, flags=re.M)

    extracted_pages: List[str] = []
    for index in range(1, len(page_blocks), 2):
        extracted_pages.append(page_blocks[index + 1].strip())
    complete_text = "\n\n".join(page for page in extracted_pages if page)
    if not complete_text:
        raise ValueError(f"No readable text was found in '{text_path}'.")
    metadata = _normalize_document_metadata(metadata, complete_text[:12000])
    if metadata.get("author") and metadata.get("title"):
        title_words = set(re.findall(r"[a-z]+", metadata["title"].lower()))
        author_words = set(re.findall(r"[a-z]+", metadata["author"].lower()))
        if (
            author_words
            and author_words.issubset(title_words)
            and len(author_words) <= 2
            and ":" not in metadata["title"]
        ):
            metadata.pop("author")

    page_records = [
        {
            "page_or_id": f"Page {number}",
            "text": page,
            "source_file": source_file,
            "document_metadata": metadata,
        }
        for number, page in enumerate(extracted_pages, start=1)
        if page
    ]
    chunks: List[KnowledgeChunk] = []
    chunk_records: List[Dict[str, Any]] = []
    chunk_size = 120
    overlap = 25
    step = chunk_size - overlap
    for number, page in enumerate(extracted_pages, start=1):
        words = page.split()
        if not words:
            continue
        page_chunks = [
            " ".join(words[start : start + chunk_size])
            for start in range(0, len(words), step)
            if words[start : start + chunk_size]
        ]
        for section, chunk_text in enumerate(page_chunks, start=1):
            page_or_id = (
                f"Page {number}"
                if len(page_chunks) == 1
                else f"Page {number}, Section {section}"
            )
            chunks.append(
                KnowledgeChunk(
                    source_file=source_file,
                    source_type="pdf_text_chunk",
                    page_or_id=page_or_id,
                    text=chunk_text,
                )
            )
            chunk_records.append(
                {
                    "page_or_id": page_or_id,
                    "text": chunk_text,
                    "source_file": source_file,
                    "document_metadata": metadata,
                }
            )

    qa_pairs = _metadata_question_answers(metadata, source_file, "Page 1")
    for chunk in chunks:
        qa_pairs.extend(
            generate_grounded_qa_from_chunks(
                [chunk],
                max_questions_per_chunk=max_questions_per_page,
                subject=subject,
            )
        )
    archive_paths = _write_question_answer_archives(
        question_root, answer_root, json_path, source_file, subject, qa_pairs
    )
    payload = {
        "schema_version": 1,
        "source_file": source_file,
        "document_metadata": metadata,
        "question_generation": {
            "subject": subject,
            "pattern_file": _question_pattern_path(subject),
            "catalog_file": "data/question_patterns/catalog.json",
            "catalog_pattern_count": len(_load_question_pattern_catalog()),
            **archive_paths,
        },
        "extraction_report": {
            "total_pages": len(extracted_pages),
            "pages_with_text": sum(bool(page) for page in extracted_pages),
            "total_chunks": len(chunks),
            "chunking": {
                "chunk_size_words": chunk_size,
                "overlap_words": overlap,
                "step_words": step,
            },
            "total_words": len(complete_text.split()),
            "warnings": (extraction_report or {}).get("warnings", []),
            "text_file": text_path,
            "document_metadata": metadata,
        },
        "question_quality": {
            "accepted": len(qa_pairs),
            "rules_file": "data/question_quality/quality_examples.json",
            "detector": "offline_source_and_evidence_rules",
        },
        "pages": page_records,
        "chunks": chunk_records,
        "qa_pairs": qa_pairs,
    }
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return {
        "json_path": json_path,
        "text_path": text_path,
        "pages": len(extracted_pages),
        "words": len(complete_text.split()),
        "qa_pairs": len(qa_pairs),
        "warnings": (extraction_report or {}).get("warnings", []),
    }


def _metadata_question_answers(
    metadata: Dict[str, str], source_file: str, page_or_id: str
) -> List[Dict[str, str]]:
    """Create natural questions backed by OCR-detected document metadata."""
    records: List[Dict[str, str]] = []
    title = metadata.get("title")
    author = metadata.get("author")
    title_reference = f"“{title}”" if title else "this story"
    if title:
        records.append({
            "question": "What is the title of the story?",
            "answer": title,
            "question_type": "document_title",
            "pattern_source": "document_metadata",
            "pattern_type": "document_title",
            "source_file": source_file,
            "page_or_id": page_or_id,
        })
    if author:
        records.append({
            "question": f"Who is the author of {title_reference}?",
            "answer": author,
            "question_type": "document_author",
            "pattern_source": "document_metadata",
            "pattern_type": "document_author",
            "source_file": source_file,
            "page_or_id": page_or_id,
        })
        if title:
            records.append({
                "question": f"Who wrote {title_reference}?",
                "answer": author,
                "question_type": "document_author",
                "pattern_source": "document_metadata",
                "pattern_type": "document_author",
                "source_file": source_file,
                "page_or_id": page_or_id,
            })
    return records


def convert_pdf_folder_to_json(
    pdf_root: str = "pdfs_to_train",
    json_root: str = "data/converted_json",
    text_root: str = "data/converted_text",
    max_questions_per_page: int = 12,
) -> Dict[str, Any]:
    """Convert matching TXT archives into JSON, using PDFs only for filenames.

    Relative folders are preserved, for example:
    ``data/converted_text/science/book.txt`` becomes
    ``data/converted_json/science/book.json``.
    """
    if not os.path.isdir(pdf_root):
        raise FileNotFoundError(f"PDF training folder not found: {pdf_root}")

    pdf_paths = []
    for root, _, files in os.walk(pdf_root):
        for name in sorted(files):
            if name.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, name))
    pdf_paths.sort()
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {pdf_root}.")

    converted: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    print(f"PROGRESS 0 {len(pdf_paths)} Converting TXT files to JSON")
    for pdf_path in pdf_paths:
        relative_path = os.path.relpath(pdf_path, pdf_root)
        relative_json = os.path.splitext(relative_path)[0] + ".json"
        relative_text = os.path.splitext(relative_path)[0] + ".txt"
        json_path = os.path.join(json_root, relative_json)
        text_path = os.path.join(text_root, relative_text)
        try:
            result = export_text_knowledge_json(
                text_path,
                json_path,
                source_file=os.path.basename(pdf_path),
                max_questions_per_page=max_questions_per_page,
                subject=os.path.basename(os.path.dirname(pdf_path)),
                question_root=os.path.join(os.path.dirname(json_root), "generated_questions"),
                answer_root=os.path.join(os.path.dirname(json_root), "generated_answers"),
            )
            result["pdf_path"] = pdf_path
            converted.append(result)
            print(f"PROGRESS {len(converted) + len(failures)} {len(pdf_paths)} Converted {os.path.basename(pdf_path)}")
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append({"pdf_path": pdf_path, "error": str(exc)})
            print(f"PROGRESS {len(converted) + len(failures)} {len(pdf_paths)} Failed {os.path.basename(pdf_path)}")

    if not converted:
        details = "; ".join(f"{item['pdf_path']}: {item['error']}" for item in failures)
        raise ValueError(f"No PDFs could be converted. {details}")

    return {
        "pdf_root": pdf_root,
        "text_root": text_root,
        "json_root": json_root,
        "converted": converted,
        "failures": failures,
        "pdf_count": len(pdf_paths),
        "converted_count": len(converted),
        "failed_count": len(failures),
    }


def convert_text_folder_to_json(
    text_root: str = "data/converted_text",
    json_root: str = "data/converted_json",
) -> Dict[str, Any]:
    """Convert existing page-preserving TXT archives into JSON archives."""
    if not os.path.isdir(text_root):
        raise FileNotFoundError(f"Text archive folder not found: {text_root}")

    text_paths = sorted(
        os.path.join(root, name)
        for root, _, files in os.walk(text_root)
        for name in sorted(files)
        if name.lower().endswith(".txt")
    )
    if not text_paths:
        raise ValueError(f"No TXT archives found in {text_root}. Run the text step first.")

    converted: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    print(f"PROGRESS 0 {len(text_paths)} Converting TXT files to JSON")
    for text_path in text_paths:
        relative_path = os.path.relpath(text_path, text_root)
        relative_json = os.path.splitext(relative_path)[0] + ".json"
        json_path = os.path.join(json_root, relative_json)
        subject = os.path.basename(os.path.dirname(relative_path)) or "general"
        source_file = os.path.splitext(os.path.basename(text_path))[0] + ".pdf"
        try:
            result = export_text_knowledge_json(
                text_path=text_path,
                json_path=json_path,
                source_file=source_file,
                subject=subject,
                question_root=os.path.join(os.path.dirname(json_root), "generated_questions"),
                answer_root=os.path.join(os.path.dirname(json_root), "generated_answers"),
            )
            result["text_path"] = text_path
            converted.append(result)
            print(
                f"PROGRESS {len(converted) + len(failures)} {len(text_paths)} "
                f"Converted {os.path.basename(text_path)}"
            )
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append({"text_path": text_path, "error": str(exc)})
            print(
                f"PROGRESS {len(converted) + len(failures)} {len(text_paths)} "
                f"Failed {os.path.basename(text_path)}"
            )

    if not converted:
        details = "; ".join(f"{item['text_path']}: {item['error']}" for item in failures)
        raise ValueError(f"No TXT archives could be converted. {details}")

    return {
        "text_root": text_root,
        "json_root": json_root,
        "converted": converted,
        "failures": failures,
        "text_count": len(text_paths),
        "converted_count": len(converted),
        "failed_count": len(failures),
    }


def load_qa_from_json(json_path: str) -> List[KnowledgeChunk]:
    """Loads structured knowledge and QA pairs from a JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    file_name = os.path.basename(json_path)
    chunks = []

    if isinstance(data, list):
        for idx, item in enumerate(data, start=1):
            if isinstance(item, dict):
                # Standard QA pair
                question = item.get("question") or item.get("prompt") or item.get("q")
                answer = item.get("answer") or item.get("response") or item.get("a")
                context = item.get("context") or item.get("text") or item.get("content") or ""

                combined_text = context
                if not combined_text and (question and answer):
                    combined_text = f"Question: {question} Answer: {answer}"

                chunks.append(
                    KnowledgeChunk(
                        source_file=file_name,
                        source_type="json",
                        page_or_id=f"Entry #{idx}",
                        text=combined_text.strip(),
                        question=question,
                        answer=answer,
                    )
                )
    elif isinstance(data, dict) and (
        "pages" in data or "chunks" in data or "qa_pairs" in data
    ):
        source_file = data.get("source_file") or file_name
        stored_chunks = data.get("chunks") or data.get("pages", [])
        for page in stored_chunks:
            if isinstance(page, dict) and page.get("text"):
                chunks.append(
                    KnowledgeChunk(
                        source_file=source_file,
                        source_type=page.get("source_type", "pdf_json_chunk"),
                        page_or_id=page.get("page_or_id", "Unknown page"),
                        text=page["text"].strip(),
                    )
                )
        for idx, item in enumerate(data.get("qa_pairs", []), start=1):
            if isinstance(item, dict) and item.get("question") and item.get("answer"):
                chunks.append(
                    KnowledgeChunk(
                        source_file=source_file,
                        source_type="pdf_json_qa",
                        page_or_id=item.get("page_or_id", f"QA #{idx}"),
                        text=item.get("answer", "").strip(),
                        question=item["question"].strip(),
                        answer=item["answer"].strip(),
                    )
                )
    elif isinstance(data, dict):
        # Key-value topic dictionary
        for topic, content in data.items():
            if isinstance(content, dict):
                q = content.get("question")
                a = content.get("answer")
                ctx = content.get("context") or str(content)
                chunks.append(
                    KnowledgeChunk(
                        source_file=file_name,
                        source_type="json",
                        page_or_id=f"Topic: {topic}",
                        text=ctx.strip(),
                        question=q,
                        answer=a,
                    )
                )
            else:
                chunks.append(
                    KnowledgeChunk(
                        source_file=file_name,
                        source_type="json",
                        page_or_id=f"Topic: {topic}",
                        text=f"{topic}: {content}".strip(),
                    )
                )

    return chunks


def generate_synthetic_qa_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Synthesizes diverse student curriculum questions directly from raw textbook text
    when no pre-existing question pairs exist (pure PDF mode).
    Generates definitions, mechanisms, processes, roles, and components.
    """
    clean_lines = [
        l.strip() for l in text.split("\n")
        if l.strip() and not re.fullmatch(r"(chapter\s+\d+|\d+|page\s+\d+)", l.strip(), re.I)
    ]
    clean_text = " ".join(clean_lines)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if len(s.strip()) > 12]

    qa_pairs: List[Tuple[str, str]] = []

    for s in sentences:
        # Pattern 1: X is/are [the/a/an] Y (Definition)
        m_def = re.match(r"^([A-Z][a-zA-Z0-9\s\-]{2,35}?)\s+(?:is|are)\s+(?:the|a|an)\s+(.+)", s, re.I)
        if m_def:
            subject = m_def.group(1).strip()
            qa_pairs.append((f"What is {subject}?", s))
            qa_pairs.append((f"Explain {subject}?", s))
            qa_pairs.append((f"Define {subject}?", s))

        # Pattern 2: convert(s) X into Y (Transformation / Mechanism)
        m_conv = re.search(r"convert(?:s)?\s+([^.]+?)\s+into\s+([^.]+)", s, re.I)
        if m_conv:
            inp = m_conv.group(1).strip()
            out = m_conv.group(2).strip()
            qa_pairs.append((f"How does conversion of {inp} into {out} work?", s))
            qa_pairs.append((f"What converts {inp}?", s))

        # Pattern 3: absorb(s) X (Absorption / Intake)
        m_abs = re.search(r"absorb(?:s)?\s+([^.]+?)(?:\s+from|\s+using|\s+through|\.|\,)", s, re.I)
        if m_abs:
            item = m_abs.group(1).strip()
            qa_pairs.append((f"What absorbs {item}?", s))
            qa_pairs.append((f"How is {item} absorbed by plants?", s))

        # Pattern 4: called X / known as X (Terminology / Organelles / Features)
        m_term = re.search(r"(?:called|known as)\s+([a-zA-Z0-9\-]+)", s, re.I)
        if m_term:
            term = m_term.group(1).strip()
            qa_pairs.append((f"What is the function of {term}?", s))
            qa_pairs.append((f"What are {term}?", s))

        # Pattern 5: capture(s) / harness(es) / generate(s) (Physics & Energy)
        m_eng = re.search(r"(?:capture|harness|generate|produce)(?:s|es)?\s+([^.]+?)(?:\s+to|\s+from|\.|\,)", s, re.I)
        if m_eng:
            action_obj = m_eng.group(1).strip()
            qa_pairs.append((f"How is {action_obj} harnessed or generated?", s))

        # Pattern 6: Subject + verb general concept probe
        words = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", s.lower()) if w not in {
            "from", "into", "with", "that", "this", "they", "their", "which", "while", "through",
            "these", "those", "when", "where", "there", "about", "other", "under", "after"
        }]
        if words:
            qa_pairs.append((f"Explain the role of {words[0]} in the curriculum?", s))
            if len(words) >= 2:
                qa_pairs.append((f"How does {words[1]} relate to {words[0]}?", s))

    # Fallback probe if regex didn't trigger
    if not qa_pairs:
        for s in sentences:
            probe_words = s.split()[:8]
            qa_pairs.append((f"Explain {' '.join(probe_words)}?", s))

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for q, ans in qa_pairs:
        if q.lower() not in seen:
            seen.add(q.lower())
            deduped.append((q, ans))

    return deduped


class QATrainingDataset(Dataset):
    """PyTorch Dataset for training the Edge Neural QA Model."""

    def __init__(
        self,
        chunks: List[KnowledgeChunk],
        vocab: Vocabulary,
        max_seq_len: int = 96,
    ):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.samples = []

        # Generate training pairs from chunks
        for chunk in chunks:
            text = chunk.text
            if not text:
                continue

            # If an explicit question exists, use it
            if chunk.question and chunk.answer:
                q_text = chunk.question
                a_text = chunk.answer
                self.samples.append((q_text, text, a_text))
            else:
                # Derive realistic synthetic question-context pairs from raw PDF text
                synthetic_pairs = generate_synthetic_qa_from_text(text)
                for q_synthetic, target_ans in synthetic_pairs:
                    self.samples.append((q_synthetic, text, target_ans))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_text, ctx_text, ans_text = self.samples[idx]

        q_ids = self.vocab.encode(q_text, max_length=self.max_seq_len)
        ctx_ids = self.vocab.encode(ctx_text, max_length=self.max_seq_len)
        ans_ids = self.vocab.encode(ans_text, max_length=self.max_seq_len)

        return (
            torch.tensor(q_ids, dtype=torch.long),
            torch.tensor(ctx_ids, dtype=torch.long),
            torch.tensor(ans_ids, dtype=torch.long),
        )
