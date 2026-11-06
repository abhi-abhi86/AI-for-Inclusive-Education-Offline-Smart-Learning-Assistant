"""
Data Loader and Preprocessor for Offline Edge Neural Model.
Extracts curriculum knowledge from PDFs (textbooks, notes) and JSON files (Q&A, syllabus),
builds local vocabulary, and creates PyTorch training datasets.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
            "This PDF appears to be scanned/image-based. "
            "Use OCR (e.g. 'ocrmypdf') to add a text layer first."
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
                        if len(chunk_words) < 20:
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

