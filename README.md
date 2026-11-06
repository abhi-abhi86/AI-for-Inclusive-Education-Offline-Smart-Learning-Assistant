<p align="center">
  <img src="assets/banner.svg" alt="Offline Smart Learning Assistant Banner" width="100%"/>
</p>

<p align="center">
  <img src="assets/logo.jpg" alt="Offline Smart Learning Assistant Logo" width="120" height="120"/>
</p>

<h1 align="center">AI for Inclusive Education: Offline Smart Learning Assistant</h1>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.15%2B-ee4c2c.svg" alt="PyTorch"/></a>
  <img src="https://img.shields.io/badge/Weights-Compressed%20.pth-blue.svg" alt="Model Format"/>
  <a href="https://ai-inclusive-education-offline-assi.vercel.app"><img src="https://img.shields.io/badge/Live%20Demo-Vercel%20Cloud-success.svg?style=flat&logo=vercel" alt="Live Demo"/></a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Network-100%25%20Offline%20(Zero%20APIs)-brightgreen.svg" alt="Offline First"/>
  <img src="https://img.shields.io/badge/Ingestion-PDF%20%7C%20JSON-orange.svg" alt="Data Sources"/>
  <img src="https://img.shields.io/badge/Platform-Raspberry%20Pi%20%7C%20macOS-purple.svg" alt="Edge AI"/>
  <img src="https://img.shields.io/badge/Model%20Size-2.2%20MB-teal.svg" alt="Model Size"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
</p>

<p align="center">
  🌐 <b>Live Hosted Web App (conversational only):</b> <a href="https://ai-inclusive-education-offline-assi.vercel.app"><b>https://ai-inclusive-education-offline-assi.vercel.app</b></a>
</p>

> **Note:** The Vercel hosted demo supports conversational chat only. PDF-based Q&A with citations requires running the app locally with PyTorch installed (see Quick Start below).

---

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Model Format](https://img.shields.io/badge/Weights-Compressed%20.pth-blue.svg)]()
[![Offline First](https://img.shields.io/badge/Network-100%25%20Offline%20(Zero%20APIs)-brightgreen.svg)]()
[![Data Sources](https://img.shields.io/badge/Ingestion-PDF%20%7C%20JSON-orange.svg)]()
[![Edge AI](https://img.shields.io/badge/Platform-Raspberry%20Pi%20%7C%20macOS-purple.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

An offline Edge AI learning companion that allows educators and students in low-connectivity regions to:
1. **Train directly on text-based PDF textbooks and notes** using native PyTorch offline backpropagation.
2. **Compress the trained model into a `.pth` file** (low memory footprint, fast edge loading, ~2–4 MB).
3. **Answer student questions with evidence-based retrieval** — answers cite the exact page and excerpt from the uploaded PDF, and explicitly say "not found" when the topic is not covered.
4. **Generate diagnostic learning aids** (3-point summaries and 3 quiz questions).

> **Scanned PDFs:** This app works with text-based PDFs only. For scanned / image-only PDFs, run OCR first: `ocrmypdf input.pdf output.pdf`

---

## 🌟 Key Capabilities

### 1. Train Purely on PDFs (No Questions Needed, No APIs)
- **Automatic Question Synthesis**: Ingests raw chapter text, extracts definitions, mechanisms, and roles, and automatically generates diverse curriculum training pairs.
- **PDF & JSON Ingestion**: Ingests standard text-based PDF textbooks or optional JSON Q&A pairs.
- **Local Training**: Uses an attention-based Transformer model (`EdgeNeuralQAModel`) with InfoNCE contrastive representation learning.
- **Compressed `.pth` File**: Packages model weights, vocabulary, and indexed knowledge embeddings into a compressed PyTorch checkpoint (`.pth`), typically **~2–4 MB** for edge deployments.
- **PDF Extraction Report**: Prints a diagnostic report for every PDF — pages with text, blank pages, and a clear warning if the PDF appears scanned.

### 2. Evidence-Based Offline Question Answering
- **Retrieval with Citations**: Retrieves the best matching passages from your uploaded PDF and cites the exact page/section.
- **Honest Confidence**: Confidence scores reflect actual retrieval quality and are not inflated. Low-confidence or off-topic questions return an explicit "not found" response instead of a fabricated answer.
- **Fast Local Inference**: Typical latency is under 5 ms on a modern CPU; actual numbers vary by hardware.
- **Separate Document & Conversation Knowledge**: PDF curriculum answers and daily conversation answers are drawn from separate collections to avoid cross-contamination.

---

## 🚀 Quick Start: Train & Answer (PyTorch `.pth` Model)

### 1. Run Automated End-to-End Demo
Creates sample science textbook (PDF) and curriculum Q&A (JSON), trains the neural model from scratch, saves `demo_curriculum_model.pth`, and queries it:
```bash
python3 train_and_qa.py demo
```

### 2. Train on Your Own PDF Textbooks and JSON Files
```bash
python3 train_and_qa.py train \
    --pdf path/to/textbook.pdf \
    --json path/to/curriculum_qa.json \
    --epochs 15 \
    --save models/my_curriculum_model.pth
```

### 3. Ask Questions Offline
```bash
python3 train_and_qa.py ask \
    --model models/my_curriculum_model.pth \
    --question "What is the function of stomata in plant leaves?"
```

Sample output:
```text
====================================================================
  Question: What is the function of stomata in plant leaves?
--------------------------------------------------------------------
  Answer:   Stomata are microscopic pores in leaves that regulate the intake of carbon dioxide and release of oxygen.

  Source:   curriculum_qa.json (Entry #2)
  Excerpt:  "Stomata on the underside of plant leaves facilitate gas exchange necessary for photosynthesis...."
--------------------------------------------------------------------
  Confidence: 87.3% | Latency: <5 ms (CPU) | Offline: 100%
====================================================================
```

### 4. Interactive Terminal Q&A Mode
Chat with your trained model offline in real time:
```bash
./venv/bin/python3 train_and_qa.py interactive --model models/demo_curriculum_model.pth
```

---

## 🖥️ Lightweight Graphical User Interface (GUI)

The application includes two lightweight GUI options with **zero external GUI dependencies**:

### Option A: Modern Lightweight Web GUI (Default)
Runs a local zero-dependency server (`http.server`) and opens directly in your browser:
```bash
./venv/bin/python3 gui.py
```
- **Input to Output workflow**:
  - **Ask Questions Tab**: Preset query pills, instant input, answer card with source page citation, confidence score (%), and latency.
  - **Train Model Tab**: Select local PDF and JSON files, adjust epochs, view real-time training metrics and compressed `.pth` size.
  - **Study Aid Tab**: Ingest any 150–300 word passage and generate structured summaries + 3 quiz questions.

### Option B: Native Desktop Window (Tkinter)
Launches a standalone native desktop window on macOS, Linux, or Raspberry Pi:
```bash
./venv/bin/python3 gui.py --desktop
```

---

## 🎯 Hackathon Deliverables (Problem Statement 1)

For the organizer-provided passage summary & quiz generation task:

```bash
# Run on curated sample passages
python3 assistant.py --sample 1   # Biology (Photosynthesis)
python3 assistant.py --sample 2   # Physics (Renewable Energy)
python3 assistant.py --sample 3   # Earth Science (Water Cycle)

# Run batch benchmark
python3 assistant.py --benchmark
```

See [`IMPACT_REPORT.md`](./IMPACT_REPORT.md) for the complete evaluation report and [`PI_SETUP.md`](./PI_SETUP.md) for the Raspberry Pi setup guide.

---

## 📁 Repository Structure

```
├── train_and_qa.py            # Primary CLI: Train on PDF/JSON & Answer offline
├── trainer.py                 # PyTorch training loop, InfoNCE loss, .pth compressor
├── model.py                   # EdgeNeuralQAModel (Transformer + Dual Semantic Heads)
├── qa_engine.py               # Local Q&A engine with Hybrid Reciprocal Rank Fusion
├── data_loader.py             # PDF (pypdf) & JSON curriculum parser + Vocabulary
├── generate_sample_data.py    # Generates sample curriculum PDF & JSON files
├── test_neural_trainer.py     # Unit tests for training, compression & inference
├── assistant.py               # Summary & Quiz CLI for hackathon evaluation
├── metrics.py                 # Telemetry (RAM RSS, latency, tok/s, 0 network calls)
├── prompt_templates.py        # Structured prompt formatters & passage validator
├── export_executorch_model.py # ExecuTorch export script for Raspberry Pi
├── IMPACT_REPORT.md           # Hackathon evaluation report answering all 5 deliverables
├── PI_SETUP.md                # Raspberry Pi 4/5 deployment guide
└── sample_data/               # Sample educational PDF and JSON datasets
```

---

## 🧪 Running Unit Tests

```bash
# Test neural training and .pth Q&A pipeline
python3 -m unittest test_neural_trainer.py

# Test hackathon educational assistant
python3 -m unittest test_assistant.py

# Test grounded PDF extraction and evidence-based Q&A (requires trained model)
python3 -m unittest tests/test_pdf_qa_grounded.py

# Run all tests at once
python3 -m unittest discover .
```

---

## 📄 License
This project is licensed under the MIT License - see [LICENSE](./LICENSE) for details.
