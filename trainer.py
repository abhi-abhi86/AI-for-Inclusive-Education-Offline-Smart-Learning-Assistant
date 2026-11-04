"""
Trainer Module for Offline Edge Neural Model.
Ingests PDF and JSON curriculum data, trains the PyTorch model locally,
compresses the weights and knowledge embeddings, and saves a .pth file.
Operates 100% offline without any API calls.
"""

import os
import time
from typing import Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    TORCH_AVAILABLE = False

from data_loader import (
    KnowledgeChunk,
    QATrainingDataset,
    Vocabulary,
    extract_text_from_pdf,
    load_qa_from_json,
)
from model import EdgeNeuralQAModel


def train_offline_model(
    pdf_paths: Optional[List[str]] = None,
    json_paths: Optional[List[str]] = None,
    save_path: str = "offline_knowledge_model.pth",
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 0.001,
    d_model: int = 128,
    num_layers: int = 3,
    device: str = "cpu",
) -> Dict:
    """
    Trains the offline neural model on supplied PDF textbooks and JSON datasets,
    compresses parameters into a .pth checkpoint, and returns training summary.
    """
    print("\n" + "=" * 68)
    print("      OFFLINE NEURAL MODEL TRAINING (PDF & JSON CURRICULUM)")
    print("=" * 68)

    all_chunks: List[KnowledgeChunk] = []

    # 1. Ingest PDF files
    if pdf_paths:
        for p in pdf_paths:
            if os.path.exists(p):
                print(f"[Data Ingestion] Extracting textbook knowledge from PDF: {os.path.basename(p)}")
                pdf_chunks = extract_text_from_pdf(p)
                all_chunks.extend(pdf_chunks)
                print(f"  -> Extracted {len(pdf_chunks)} contextual knowledge chunks.")
            else:
                print(f"[Warning] PDF file not found: {p}")

    # 2. Ingest JSON files
    if json_paths:
        for j in json_paths:
            if os.path.exists(j):
                print(f"[Data Ingestion] Loading structured Q&A / notes from JSON: {os.path.basename(j)}")
                json_chunks = load_qa_from_json(j)
                all_chunks.extend(json_chunks)
                print(f"  -> Loaded {len(json_chunks)} Q&A / curriculum entries.")
            else:
                print(f"[Warning] JSON file not found: {j}")

    if not all_chunks:
        raise ValueError("No valid knowledge chunks extracted. Please provide at least one valid PDF or JSON file.")

    print(f"\n[Knowledge Base] Total curriculum chunks: {len(all_chunks)}")

    # 3. Build Vocabulary
    print("[Vocabulary] Constructing local vocabulary from documents...")
    vocab = Vocabulary(min_freq=1, max_size=8000)
    all_texts = [c.text for c in all_chunks]
    for c in all_chunks:
        if c.question:
            all_texts.append(c.question)
        if c.answer:
            all_texts.append(c.answer)

    vocab.build_from_texts(all_texts)
    print(f"  -> Built local vocabulary size: {len(vocab)} unique tokens")

    # 4. Create PyTorch Dataset & DataLoader
    dataset = QATrainingDataset(all_chunks, vocab=vocab, max_seq_len=96)
    if len(dataset) == 0:
        raise ValueError("Training dataset generated 0 samples. Verify input text length.")

    data_loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    print(f"[Dataset] Generated {len(dataset)} training QA pairs across {len(data_loader)} batches.")

    # 5. Initialize Model
    dev = torch.device(device)
    model = EdgeNeuralQAModel(
        vocab_size=len(vocab),
        d_model=d_model,
        nhead=4,
        num_layers=num_layers,
        dim_feedforward=d_model * 2,
        embedding_dim=64,
        pad_idx=vocab.pad_id,
        dropout=0.1,
    ).to(dev)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model Architecture] Initialized EdgeNeuralQAModel with {total_params:,} parameters.")

    # 6. Optimizer & Loss Functions
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    temperature = 0.1

    print("\n[Training Loop] Beginning local backpropagation (100% Offline, Zero APIs)...")
    start_time = time.time()

    model.train()
    history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_retrieval_loss = 0.0
        epoch_ans_loss = 0.0

        for q_batch, ctx_batch, ans_batch in data_loader:
            q_batch = q_batch.to(dev)
            ctx_batch = ctx_batch.to(dev)
            ans_batch = ans_batch.to(dev)
            batch_sz = q_batch.size(0)

            optimizer.zero_grad()
            outputs = model(q_batch, ctx_batch)

            q_vec = outputs["q_vector"]      # [B, dim]
            ctx_vec = outputs["ctx_vector"]  # [B, dim]
            ans_logits = outputs["answer_logits"]

            # InfoNCE In-Batch Negative Contrastive Loss
            # Similarity matrix between all queries and all contexts in the batch
            sim_matrix = torch.matmul(q_vec, ctx_vec.T) / temperature  # [B, B]
            labels = torch.arange(batch_sz, device=dev)  # Diagonal pairs are positives
            loss_retrieval = F.cross_entropy(sim_matrix, labels)

            # Target 2: Answer token cross-entropy prediction
            target_first_token = ans_batch[:, 1]  # index 0 is <bos>
            loss_answer = ce_loss_fn(ans_logits, target_first_token)

            loss = loss_retrieval + 0.3 * loss_answer
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_retrieval_loss += loss_retrieval.item()
            epoch_ans_loss += loss_answer.item()

        avg_loss = epoch_loss / len(data_loader)
        history.append(avg_loss)
        if epoch == 1 or epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Loss: {avg_loss:.4f} (Retrieval: {epoch_retrieval_loss / len(data_loader):.4f}, Ans: {epoch_ans_loss / len(data_loader):.4f})")

    elapsed_time = time.time() - start_time
    print(f"\n[Training Complete] Finished {epochs} epochs in {elapsed_time:.2f} seconds.")

    # 7. Pre-compute and Cache Neural Knowledge Embeddings for Fast Edge Inference
    print("[Compression & Indexing] Pre-computing normalized neural embeddings for all knowledge chunks...")
    model.eval()
    knowledge_embeddings = []
    with torch.no_grad():
        for chunk in all_chunks:
            ids = torch.tensor([vocab.encode(chunk.text, max_length=96)], dtype=torch.long, device=dev)
            vec = model.encode_semantic_vector(ids)
            knowledge_embeddings.append(vec.cpu().squeeze(0))

    stacked_embeddings = torch.stack(knowledge_embeddings)  # [num_chunks, embedding_dim]

    # 8. Save Compressed .pth Checkpoint
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab.to_dict(),
        "config": model.config,
        "knowledge_chunks": [
            {
                "source_file": c.source_file,
                "source_type": c.source_type,
                "page_or_id": c.page_or_id,
                "text": c.text,
                "question": c.question,
                "answer": c.answer,
            }
            for c in all_chunks
        ],
        "knowledge_embeddings": stacked_embeddings,
        "training_stats": {
            "epochs": epochs,
            "final_loss": history[-1],
            "total_params": total_params,
            "training_time_s": elapsed_time,
            "num_chunks": len(all_chunks),
            "sources": (pdf_paths or []) + (json_paths or []),
        },
    }

    torch.save(checkpoint, save_path)
    file_size_mb = os.path.getsize(save_path) / (1024.0 * 1024.0)

    print("=" * 68)
    print(f"  Saved Compressed Model File: {save_path}")
    print(f"  File Size on Disk:           {file_size_mb:.2f} MB")
    print(f"  Knowledge Passages Indexed:  {len(all_chunks)}")
    print(f"  Offline Capability:          100% (No APIs, No Cloud)")
    print("=" * 68 + "\n")

    return checkpoint["training_stats"]
