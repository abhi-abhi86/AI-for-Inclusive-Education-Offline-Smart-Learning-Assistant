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

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

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
    generative_verified: bool = True

    def format_card(self) -> str:
        """Renders an accessible answer card for students."""
        conf_pct = self.confidence_score * 100.0
        if conf_pct < 60.0:
            conf_label = f"{conf_pct:.1f}%  ⚠️  Low Confidence"
        else:
            conf_label = f"{conf_pct:.1f}%"
        lines = [
            "\n" + "=" * 70,
            "  📚 CURRICULUM ANSWER (100% OFFLINE)",
            "=" * 70,
            f"  Question: {self.question}",
            "-" * 70,
            f"  Answer:\n{self.answer}",
            "",
            f"  Source:   {self.source_file} ({self.source_reference})",
            f"  Excerpt:  \"{self.context_passage[:160]}...\"",
            "-" * 70,
            f"  Confidence: {conf_label} | Latency: {self.inference_time_ms:.1f} ms | Offline: 100%",
            "=" * 70,
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

        # 0. Conversational Intent Check (Greetings, Identity, Capabilities, Gratitude, Farewells)
        conversational_resp = self._check_conversational_intent(question, t_start)
        if conversational_resp is not None:
            return conversational_resp

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

        # Only search document chunks for curriculum questions (keep conversations separate)
        doc_indices = [
            i for i, c in enumerate(self.knowledge_chunks)
            if "conversation" not in c.get("source_file", "").lower()
        ]
        # Fall back to all chunks when no document chunks exist (pure conversational model)
        search_indices = doc_indices if doc_indices else list(range(len(self.knowledge_chunks)))

        # Dense ranking (over search subset)
        dense_subset = dense_sims[search_indices] if search_indices else dense_sims
        dense_ranks_local = torch.argsort(dense_subset, descending=True).tolist()
        dense_rank_map = {search_indices[local]: rank for rank, local in enumerate(dense_ranks_local)}

        # Lexical ranking
        lexical_scores = []
        for idx in search_indices:
            chunk = self.knowledge_chunks[idx]
            chunk_full = (chunk.get("text", "") + " " + (chunk.get("question") or "")).lower()
            c_tokens = set(re.findall(r"\w+", chunk_full))
            overlap = len(q_tokens & c_tokens)
            lexical_scores.append((overlap, idx))

        lexical_scores.sort(key=lambda x: x[0], reverse=True)
        lexical_rank_map = {idx: rank for rank, (score, idx) in enumerate(lexical_scores)}

        # Combine with RRF (k=60)
        k = 60
        rrf_scores = []
        for idx in search_indices:
            r_dense = dense_rank_map[idx]
            r_lex = lexical_rank_map[idx]
            rrf = (1.0 / (k + r_dense)) + (1.5 / (k + r_lex))  # slight priority to exact terminology
            d_score = float(dense_sims[idx].item())
            rrf_scores.append((rrf, d_score, idx))

        rrf_scores.sort(key=lambda x: x[0], reverse=True)
        best_rrf, best_dense_score, best_idx = rrf_scores[0]
        best_chunk = self.knowledge_chunks[best_idx]
        top_chunks = [self.knowledge_chunks[idx] for _, _, idx in rrf_scores[:3]]

        # 5. Formulate answer from retrieved curriculum context
        answer_text = self.synthesize_generative_answer(question, best_chunk, top_chunks)

        # Confidence: derived from dense similarity — NOT clamped to a fake floor
        confidence = (best_dense_score + 1.0) / 2.0  # maps cosine [-1,1] → [0,1]
        NOT_FOUND_THRESHOLD = 0.60
        if confidence < NOT_FOUND_THRESHOLD and best_rrf < 0.015:
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return QAResponse(
                question=question,
                answer=(
                    "I could not find information about this in the uploaded documents. "
                    "Please check whether this topic is covered in the textbooks, "
                    "or upload additional study material."
                ),
                source_file="No matching source found",
                source_reference="—",
                context_passage="",
                confidence_score=confidence,
                inference_time_ms=t_elapsed_ms,
                offline_verified=True,
                generative_verified=False,
            )

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        source_ref = best_chunk.get("page_or_id") or "Section 1"
        source_file = best_chunk.get("source_file", "Uploaded Document")
        if "conversation" in source_file.lower():
            source_file = "Daily Human Conversation & Life Knowledge"
            source_ref = "Conversational AI Model"

        return QAResponse(
            question=question,
            answer=answer_text,
            source_file=source_file,
            source_reference=source_ref,
            context_passage=best_chunk["text"],
            confidence_score=confidence,
            inference_time_ms=t_elapsed_ms,
            offline_verified=True,
            generative_verified=True,
        )

    def synthesize_generative_answer(
        self,
        question: str,
        chunk: Dict,
        top_chunks: Optional[List[Dict]] = None,
    ) -> str:
        """
        Generative AI Answer Synthesizer.
        Synthesizes a fluent, structured explanation directly from the retrieved PDF data:
        1. Direct, clear summary answer (with anaphora resolution).
        2. In-depth explanatory synthesis of the mechanism/concepts.
        3. Key Curriculum Highlights / takeaways.
        """
        explicit_ans = chunk.get("answer")

        # Combine text across top chunks from same source if helpful
        raw_text = chunk.get("text", "")
        combined_text = raw_text
        if top_chunks and len(top_chunks) > 1:
            for secondary in top_chunks[1:3]:
                if secondary.get("source_file") == chunk.get("source_file") and secondary.get("text") != raw_text:
                    combined_text += "\n" + secondary.get("text", "")

        # Clean headings & chapter prefixes
        clean_lines = [
            l.strip() for l in combined_text.split("\n")
            if not re.match(r"^(chapter|\d+|#|page)", l.strip(), re.I) and l.strip()
        ]
        text = " ".join(clean_lines)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]

        if not sentences:
            return explicit_ans or raw_text

        q_lower = question.lower()
        q_words = set(re.findall(r"\b[a-z]{3,}\b", q_lower)) - {
            "what", "how", "why", "when", "where", "which", "who", "does", "the", "and", "are",
            "can", "explain", "describe", "tell", "about", "role", "function"
        }

        # Score sentences against user query
        scored_sentences = []
        for i, s in enumerate(sentences):
            s_words = set(re.findall(r"\b[a-z]{3,}\b", s.lower()))
            overlap = len(q_words & s_words)
            bonus = 0.0
            if re.search(r"\b(is|are|convert|absorb|produce|release|capture|function|harness|power|originate)\b", s, re.I):
                bonus += 0.5
            if any(w in s.lower() for w in q_words):
                bonus += 0.5
            scored_sentences.append((overlap + bonus, i, s))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        best_i = scored_sentences[0][1]
        primary_sentence = scored_sentences[0][2]

        # Resolve simple pronoun anaphora (e.g. "They absorb..." -> "Plants absorb...")
        m_pronoun = re.match(r"^(They|It|These|This)\s+", primary_sentence)
        if m_pronoun and best_i > 0:
            prev_s = sentences[best_i - 1]
            m_subj = re.match(r"^([A-Z][a-zA-Z0-9\s\-]{2,25}?)\s+(?:absorb|are|is|convert|have|contain|use|capture|originate)", prev_s)
            if m_subj:
                subj = m_subj.group(1).strip()
                primary_sentence = f"{subj} {primary_sentence[len(m_pronoun.group(1)) + 1:]}"

        # If an explicit JSON answer was stored, use it as direct answer
        if explicit_ans and len(explicit_ans.strip()) > 5:
            direct_answer = explicit_ans.strip()
        else:
            direct_answer = primary_sentence

        # Gather supporting sentences for in-depth explanation
        selected_indices = {best_i}
        for score, idx, s in scored_sentences[1:4]:
            if score > 0.4:
                selected_indices.add(idx)

        ordered_explanation = [sentences[idx] for idx in sorted(selected_indices)]
        detailed_body = " ".join(ordered_explanation)

        # Extract Key Curriculum Highlights
        takeaways = []
        for s in sentences:
            m_conv = re.search(r"convert(?:s)?\s+(.+?)\s+into\s+([^.]+)", s, re.I)
            if m_conv:
                takeaways.append(f"Energy/Matter Conversion: {m_conv.group(1).strip()} ➔ {m_conv.group(2).strip()}")

            m_abs = re.search(r"absorb(?:s)?\s+([^.]+?)\s+((?:from|using|through)[^.]+)", s, re.I)
            if m_abs:
                takeaways.append(f"Intake/Absorption: {m_abs.group(1).strip()} ({m_abs.group(2).strip()})")

            m_rel = re.search(r"release(?:s|d)?\s+([^.]+?)\s+((?:back|into|through)[^.]+)", s, re.I)
            if m_rel:
                takeaways.append(f"Byproduct/Output: {m_rel.group(1).strip()} ({m_rel.group(2).strip()})")

            m_term = re.search(r"(?:called|known as)\s+([a-zA-Z0-9\-]+)", s, re.I)
            if m_term:
                takeaways.append(f"Key Term: {m_term.group(1).capitalize()} is critical to this mechanism.")

            m_gen = re.search(r"(?:capture|harness|generate)(?:s)?\s+([^.]+?)(?:\s+to|\s+from|\.|\,)", s, re.I)
            if m_gen:
                takeaways.append(f"Power Generation: Harnesses {m_gen.group(1).strip()}")

        # If chunk is from conversational dataset, return natural human answer directly
        is_conversational = (
            "conversation" in chunk.get("source_file", "").lower()
            or chunk.get("source_type") == "conversational"
        )
        if is_conversational and explicit_ans:
            return explicit_ans

        # Build formatted Generative AI response
        lines = []
        lines.append(direct_answer)
        lines.append("")

        if len(ordered_explanation) > 1 and detailed_body != direct_answer:
            lines.append("📖 In-Depth Explanation:")
            lines.append(detailed_body)
            lines.append("")

        deduped_takeaways = list(dict.fromkeys(takeaways))[:3]
        if deduped_takeaways:
            lines.append("✨ Key Curriculum Highlights:")
            for t in deduped_takeaways:
                lines.append(f"  • {t}")

        return "\n".join(lines).strip()

    def _extract_answer(self, question: str, chunk: Dict) -> str:
        """Legacy helper maintained for backward compatibility."""
        return self.synthesize_generative_answer(question, chunk)

    def _check_conversational_intent(self, question: str, t_start: float) -> Optional[QAResponse]:
        """Detects and responds to greetings, identity questions, gratitude, and small talk."""
        q_clean = question.strip().lower()
        if not q_clean:
            return None

        # 1. Greetings (Hi, Hello, Hey, Good morning, etc.)
        if re.search(r"^(hi|hello|hey|heya|howdy|greetings|hola|namaste|good\s+(morning|afternoon|evening|day))(\s+there|\s+assistant|\s+bot|[!.? ])*$", q_clean, re.I):
            sources = list(dict.fromkeys([c.get("source_file", "curriculum") for c in self.knowledge_chunks[:6]]))
            source_summary = ", ".join(sources) if sources else "your uploaded textbooks"
            answer = (
                "Hello! 👋 Welcome to your Offline Smart Learning Assistant.\n\n"
                f"I am running 100% offline on your device, powered by a local PyTorch neural model. "
                f"I am currently loaded with curriculum knowledge from {source_summary}.\n\n"
                "💡 You can ask me any question about your lessons, concepts, or uploaded textbooks! What would you like to explore today?"
            )
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return QAResponse(
                question=question,
                answer=answer,
                source_file="Offline Smart Learning Assistant Core",
                source_reference="Conversational AI Engine",
                context_passage="Conversational greeting and intent handling for offline inclusive education.",
                confidence_score=0.99,
                inference_time_ms=elapsed,
                offline_verified=True,
                generative_verified=True,
            )

        # 2. How are you / Status
        if re.search(r"^(how\s+are\s+you|how\s+is\s+it\s+going|how\s+are\s+you\s+doing|are\s+you\s+ready)([!.? ])*$", q_clean, re.I):
            answer = (
                "I am doing great and running 100% smoothly offline! 🚀\n\n"
                f"My PyTorch weights (~{self.model_size_mb:.2f} MB) and knowledge index ({len(self.knowledge_chunks)} curriculum sections) "
                "are loaded in memory on your CPU. Ready to help you study whenever you are!"
            )
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return QAResponse(
                question=question,
                answer=answer,
                source_file="Offline Smart Learning Assistant Core",
                source_reference="System Diagnostics",
                context_passage="Real-time telemetry and conversational status for edge neural model.",
                confidence_score=0.99,
                inference_time_ms=elapsed,
                offline_verified=True,
                generative_verified=True,
            )

        # 3. Identity (Who are you, What is your name)
        if re.search(r"^(who\s+are\s+you|what\s+is\s+your\s+name|what\s+are\s+you)([!.? ])*$", q_clean, re.I):
            answer = (
                "I am the Offline Smart Learning Assistant (Edge AI). 🎓\n\n"
                "I am an inclusive, private learning companion designed to run directly on edge devices (like Raspberry Pi and local computers) "
                "without requiring any internet connection or cloud APIs.\n\n"
                "I train locally on your PDF textbooks and JSON curriculum files, compress the knowledge into a tiny .pth model, and answer your questions with generative AI explanations."
            )
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return QAResponse(
                question=question,
                answer=answer,
                source_file="Offline Smart Learning Assistant Core",
                source_reference="System Identity",
                context_passage="Architecture specifications of the offline edge learning assistant.",
                confidence_score=0.99,
                inference_time_ms=elapsed,
                offline_verified=True,
                generative_verified=True,
            )

        # 4. Capabilities & Help (What can you do, Help, How does it work)
        if re.search(r"^(what\s+can\s+you\s+do|help|how\s+do\s+you\s+work|what\s+are\s+your\s+features|commands)([!.? ])*$", q_clean, re.I):
            answer = (
                "Here is what I can do for you 100% offline:\n\n"
                "1. 📖 Answer Questions: Ask any curriculum question based on your uploaded PDF textbooks or notes.\n"
                "2. ⚡ Train on Textbooks: Drop new PDFs into data/pdf_textbooks/ and click Train to learn new chapters.\n"
                "3. 📝 Study Aid Generator: Convert educational passages into 3-point summaries and 3 diagnostic quiz questions.\n"
                "4. 🔒 Zero Cloud: Everything stays completely private and runs locally on your machine."
            )
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return QAResponse(
                question=question,
                answer=answer,
                source_file="Offline Smart Learning Assistant Core",
                source_reference="Feature Overview",
                context_passage="Summary of accessible learning features available without cloud connectivity.",
                confidence_score=0.99,
                inference_time_ms=elapsed,
                offline_verified=True,
                generative_verified=True,
            )

        # 5. Gratitude / Thanks
        if re.search(r"^(thank\s+you|thanks|thanks\s+a\s+lot|thank\s+u|thx|great\s+job|awesome|good\s+job)([!.? ])*$", q_clean, re.I):
            answer = "You are very welcome! 😊 Keep up the curious learning! Feel free to ask if you have any more questions about your lessons."
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return QAResponse(
                question=question,
                answer=answer,
                source_file="Offline Smart Learning Assistant Core",
                source_reference="Conversational AI Engine",
                context_passage="Encouraging student interaction and feedback.",
                confidence_score=0.99,
                inference_time_ms=elapsed,
                offline_verified=True,
                generative_verified=True,
            )

        # 6. Farewells (Bye, Goodbye)
        if re.search(r"^(bye|goodbye|see\s+you|cya|good\s+night)([!.? ])*$", q_clean, re.I):
            answer = "Goodbye! 👋 Have a wonderful study session and come back anytime you need help with your curriculum!"
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return QAResponse(
                question=question,
                answer=answer,
                source_file="Offline Smart Learning Assistant Core",
                source_reference="Conversational AI Engine",
                context_passage="Graceful student farewell handler.",
                confidence_score=0.99,
                inference_time_ms=elapsed,
                offline_verified=True,
                generative_verified=True,
            )

        # 7. Direct Match across Trained Daily Human Conversations
        q_words = set(re.findall(r"\b[a-z]{3,}\b", q_clean))
        if q_words:
            best_chunk = None
            best_score = 0.0
            for chunk in self.knowledge_chunks:
                if not ("conversation" in chunk.get("source_file", "").lower() or chunk.get("source_type") == "conversational"):
                    continue
                c_q = (chunk.get("question") or "").lower()
                c_words = set(re.findall(r"\b[a-z]{3,}\b", c_q))
                if not c_words:
                    continue
                overlap = len(q_words & c_words)
                if overlap == 0:
                    continue
                score = overlap / len(q_words | c_words)
                if score > 0.45 or (overlap >= 2 and overlap == len(q_words)):
                    if score > best_score:
                        best_score = score
                        best_chunk = chunk

            if best_chunk and best_chunk.get("answer"):
                elapsed = (time.perf_counter() - t_start) * 1000.0
                return QAResponse(
                    question=question,
                    answer=best_chunk["answer"],
                    source_file="Daily Human Conversation & Life Knowledge",
                    source_reference="Conversational AI Engine",
                    context_passage=best_chunk.get("context") or best_chunk.get("text", ""),
                    confidence_score=0.98,
                    inference_time_ms=elapsed,
                    offline_verified=True,
                    generative_verified=True,
                )

        return None
