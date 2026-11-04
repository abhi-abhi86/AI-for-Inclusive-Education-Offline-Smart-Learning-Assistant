"""
Data Loader and Preprocessor for Offline Edge Neural Model.
Extracts curriculum knowledge from PDFs (textbooks, notes) and JSON files (Q&A, syllabus),
builds local vocabulary, and creates PyTorch training datasets.
"""

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

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


def extract_text_from_pdf(pdf_path: str, chunk_size: int = 120, overlap: int = 25) -> List[KnowledgeChunk]:
    """Extracts text from PDF page by page and chunks into overlapping passages."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    chunks = []
    file_name = os.path.basename(pdf_path)

    if PYPDF_AVAILABLE:
        reader = pypdf.PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if not page_text:
                continue

            # Chunk into paragraphs / sentences
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
                step = chunk_size - overlap
                for i in range(0, len(words), step):
                    chunk_words = words[i : i + chunk_size]
                    if len(chunk_words) < 20 and chunks:
                        # Append small leftovers to previous chunk
                        continue
                    chunk_text = " ".join(chunk_words)
                    chunks.append(
                        KnowledgeChunk(
                            source_file=file_name,
                            source_type="pdf",
                            page_or_id=f"Page {page_num} (Section {i // step + 1})",
                            text=chunk_text,
                        )
                    )
    else:
        # Fallback reading raw binary stream text
        with open(pdf_path, "rb") as fp:
            content = fp.read().decode("latin-1", errors="ignore")
        # Extract plain readable strings
        text_matches = re.findall(r"\((.*?)\)T[jJ]", content)
        raw_text = " ".join(text_matches).strip()
        if raw_text:
            chunks.append(
                KnowledgeChunk(
                    source_file=file_name,
                    source_type="pdf",
                    page_or_id="Extracted Stream",
                    text=raw_text,
                )
            )

    return chunks


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
                # Derive synthetic question-context pairs from sentences
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
                for s in sentences:
                    # Synthetic probe query
                    probe_words = s.split()[:8]
                    q_synthetic = f"Explain {' '.join(probe_words)}?"
                    self.samples.append((q_synthetic, text, s))

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

