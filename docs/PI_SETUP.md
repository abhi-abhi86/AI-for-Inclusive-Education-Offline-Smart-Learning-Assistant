# Raspberry Pi Deployment Guide: ExecuTorch Edge AI Assistant

This guide outlines the step-by-step procedure to deploy and run the **Offline Smart Learning Assistant** on a **Raspberry Pi 4** or **Raspberry Pi 5** running 64-bit Raspberry Pi OS.

---

## 1. Hardware & System Prerequisites

- **Board**: Raspberry Pi 4 Model B (4GB/8GB RAM recommended) or Raspberry Pi 5.
- **OS**: Raspberry Pi OS (64-bit / Debian Bookworm).
- **Storage**: MicroSD card (Class 10 / A2) with at least 16GB free storage.
- **Connectivity**: Internet connection required *only once* during setup to install dependencies; zero connectivity required thereafter.

---

## 2. System Dependency Installation

On your Raspberry Pi terminal, install essential C++ build tools and Python packages:

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    python3-venv \
    libomp-dev
```

---

## 3. Clone Repository & Setup Python Environment

```bash
git clone https://github.com/abhi/AI-for-Inclusive-Education-Offline-Smart-Learning-Assistant.git
cd AI-for-Inclusive-Education-Offline-Smart-Learning-Assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

---

## 4. Building ExecuTorch with XNNPACK on Raspberry Pi

ExecuTorch provides optimized ARM NEON CPU acceleration via the XNNPACK backend.

```bash
# Clone ExecuTorch
git clone --recursive https://github.com/pytorch/executorch.git
cd executorch

# Install Python requirements
./install_requirements.sh

# Configure and build the Llama runner for ARM64 CPU
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DEXECUTORCH_BUILD_XNNPACK=ON \
      -DEXECUTORCH_BUILD_LLAMA=ON \
      ..

cmake --build . --target llama_runner -j4
sudo cp bin/llama_runner /usr/local/bin/
cd ../..
```

---

## 5. Exporting or Copying Model Weights

Transfer the exported 4-bit quantized model (`llama3_2_1b.pte`) and tokenizer:

```bash
# Export using helper script
python3 tools/export_executorch_model.py --model-id "meta-llama/Llama-3.2-1B-Instruct" --quant xnnpack-4bit

# Alternatively, copy pre-exported .pte file from your workstation:
scp workstation:~/models/llama3_2_1b.pte ./models/
scp workstation:~/models/tokenizer.bin ./models/
```

---

## 6. Running the Offline Assistant on Raspberry Pi

### A. Instant Verification / Built-in Edge Engine
Run without external binaries or prior to building ExecuTorch:
```bash
# Run on sample curriculum passage (Photosynthesis)
python3 -m tools.assistant --sample 1

# Run on custom text file
python3 -m tools.assistant --file sample_passages/02_renewable_energy.txt

# Run automated batch benchmark
python3 -m tools.assistant --benchmark
```

### B. With Compiled ExecuTorch Binary
```bash
python3 -m tools.assistant \
    --executorch \
    --model-path ./models/llama3_2_1b.pte \
    --tokenizer-path ./models/tokenizer.bin \
    --runner-bin /usr/local/bin/llama_runner \
    --file sample_passages/01_photosynthesis.txt
```

---

## 7. Performance & Resource Expectations on Raspberry Pi

| Device | Model | Quantization | RAM Usage | Speed (tok/s) | Network Calls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raspberry Pi 5 (8GB)** | Llama 3.2 1B | 4-bit XNNPACK | ~1,120 MB | 14 – 18 tok/s | **0 (Offline)** |
| **Raspberry Pi 4 (4GB)** | Llama 3.2 1B | 4-bit XNNPACK | ~1,150 MB | 8 – 11 tok/s | **0 (Offline)** |
| **Raspberry Pi 4/5** | Built-in Engine | Low Footprint | ~21 MB | Real-time | **0 (Offline)** |
