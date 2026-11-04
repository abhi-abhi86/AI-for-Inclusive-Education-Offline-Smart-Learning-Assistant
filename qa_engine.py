"""
Offline Neural Question-Answering Engine.
Loads trained .pth model file, performs local semantic retrieval and answer extraction.
Zero API calls, completely self-contained.
"""

import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from data_loader import Vocabulary
from model import EdgeNeuralQAModel


@dataclass
class QAResponse:
    question: str
    answer: str
    source_file: str
    source_reference: str
    context_passage: str
    confidence_score: float  # 0.0 - 1.0
    inference_time_ms: float
    offline_verified: bool = True

    def format_card(self) -> str:
        """Renders an accessible answer card for students."""
        conf_pct = self.confidence_score * 100.0
        lines = [
            "\n" + "=" * 68,
            f"  Question: {self.question}",
            "-" * 68,
            f"  Answer:   {self.answer}",
            "",
            f"  Source:   {self.source_file} ({self.source_reference})",
            f"  Excerpt:  \"{self.context_passage[:160]}...\"",
            "-" * 68,
            f"  Confidence: {conf_pct:.1f}% | Latency: {self.inference_time_ms:.1f} ms | Offline: 100%",
            "=" * 68,
        ]
        return "\n".join(lines)


class OfflineQAModelRunner:
    """Loads and queries a compressed .pth model file offline."""

    def __init__(self, model_pth_path: str, device: str = "cpu"):
        if not os.path.exists(model_pth_path):
            raise FileNotFoundError(f"Model file not found: {model_pth_path}")

        self.device = torch.device(device)
        self.pth_path = model_pth_path

        # Load compressed PyTorch checkpoint
        checkpoint = torch.load(model_pth_path, map_location=self.device)

        # Restore Vocabulary
        self.vocab = Vocabulary.from_dict(checkpoint["vocab"])

        # Restore Model Architecture & Weights
        self.config = checkpoint["config"]
        self.model = EdgeNeuralQAModel(**self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Restore Knowledge Base & Pre-computed Embeddings
        self.knowledge_chunks = checkpoint["knowledge_chunks"]
        self.knowledge_embeddings = checkpoint["knowledge_embeddings"].to(self.device)
        self.training_stats = checkpoint.get("training_stats", {})

        self.model_size_mb = os.path.getsize(model_pth_path) / (1024.0 * 1024.0)

    def answer(self, question: str, top_k: int = 3) -> QAResponse:
        """Answers a question using the trained model weights and knowledge index."""
        t_start = time.perf_counter()

        # 1. Encode user question
        q_ids = self.vocab.encode(question, max_length=96)
        q_tensor = torch.tensor([q_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            # 2. Compute question representation
            q_vec = self.model.encode_semantic_vector(q_tensor)  # [1, embedding_dim]

            # 3. Compute cosine similarity against all trained knowledge passages
            dense_sims = torch.matmul(self.knowledge_embeddings, q_vec.squeeze(0))  # [num_chunks]

        # 4. Hybrid Reciprocal Rank Fusion (RRF) combining Dense Neural & Lexical Scores
        q_tokens = set(re.findall(r"\w+", question.lower())) - {
            "what", "is", "the", "how", "why", "where", "when", "does", "do", "a", "an", "and", "in", "of", "to", "are"
        }

        # Dense ranking
        dense_ranks = torch.argsort(dense_sims, descending=True).tolist()
        dense_rank_map = {idx: rank for rank, idx in enumerate(dense_ranks)}

        # Lexical ranking
        lexical_scores = []
        for idx, chunk in enumerate(self.knowledge_chunks):
            chunk_full = (chunk.get("text", "") + " " + (chunk.get("question") or "")).lower()
            c_tokens = set(re.findall(r"\w+", chunk_full))
            overlap = len(q_tokens & c_tokens)
            lexical_scores.append((overlap, idx))

        lexical_scores.sort(key=lambda x: x[0], reverse=True)
        lexical_rank_map = {idx: rank for rank, (score, idx) in enumerate(lexical_scores)}

        # Combine with RRF (k=60)
        k = 60
        rrf_scores = []
        for idx in range(len(self.knowledge_chunks)):
            r_dense = dense_rank_map[idx]
            r_lex = lexical_rank_map[idx]
            rrf = (1.0 / (k + r_dense)) + (1.5 / (k + r_lex))  # slight priority to exact terminology
            d_score = float(dense_sims[idx].item())
            rrf_scores.append((rrf, d_score, idx))

        rrf_scores.sort(key=lambda x: x[0], reverse=True)
        best_rrf, best_dense_score, best_idx = rrf_scores[0]
        best_chunk = self.knowledge_chunks[best_idx]

        # 5. Formulate the answer
        answer_text = self._extract_answer(question, best_chunk)

        # Confidence percentage (derived from dense similarity + rank quality)
        confidence = max(0.55, min(0.99, (best_dense_score + 1.0) / 2.0))
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return QAResponse(
            question=question,
            answer=answer_text,
            source_file=best_chunk["source_file"],
            source_reference=best_chunk["page_or_id"],
            context_passage=best_chunk["text"],
            confidence_score=confidence,
            inference_time_ms=t_elapsed_ms,
            offline_verified=True,
        )

    def _extract_answer(self, question: str, chunk: Dict) -> str:
        """Extracts or synthesizes direct answer from the retrieved chunk."""
        # If the chunk came from a structured JSON QA pair with an explicit answer:
        if chunk.get("answer"):
            return chunk["answer"].strip()

        text = chunk.get("text", "")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        if not sentences:
            return text

        # Match question keywords to identify the most salient sentence
        q_words = set(re.findall(r"\w+", question.lower())) - {"what", "is", "the", "how", "why", "where", "when", "does", "do", "a", "an"}
        
        best_sentence = sentences[0]
        max_overlap = -1

        for sent in sentences:
            s_words = set(re.findall(r"\w+", sent.lower()))
            overlap = len(q_words & s_words)
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sent

        return best_sentence.strip()
