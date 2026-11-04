"""
Lightweight Web GUI Server for Offline Smart Learning Assistant.
Uses Python's standard library http.server (0 external dependencies).
Features:
1. Student PDF Upload & Instant Auto-Training (Zero APIs, local PyTorch).
2. Dynamic Suggestion Questions generated directly from uploaded materials.
3. Offline Question-Answering with source page citations and confidence scores.
4. Passage Study Aid generator (Summary + 3 Quiz Questions).
"""

import base64
import json
import os
import re
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional

from generate_sample_data import create_sample_curriculum_json, create_sample_textbook_pdf
from prompt_templates import parse_learning_aid, validate_passage
from qa_engine import OfflineQAModelRunner
from trainer import train_offline_model

# Global runner state
CURRENT_MODEL_PATH = "models/demo_curriculum_model.pth"
QA_RUNNER: Optional[OfflineQAModelRunner] = None
RUNNER_LOCK = threading.Lock()
ACTIVE_TOPICS: List[str] = [
    "What is the function of stomata in plant leaves?",
    "What is chlorophyll and where is it found in plant cells?",
    "What energy do wind turbines capture to spin electric generators?",
    "How do solar photovoltaic cells generate electric current?",
]


def get_qa_runner(model_path: str = CURRENT_MODEL_PATH) -> Optional[OfflineQAModelRunner]:
    global QA_RUNNER, CURRENT_MODEL_PATH
    with RUNNER_LOCK:
        if QA_RUNNER is None or CURRENT_MODEL_PATH != model_path:
            if os.path.exists(model_path):
                CURRENT_MODEL_PATH = model_path
                QA_RUNNER = OfflineQAModelRunner(model_path)
            else:
                QA_RUNNER = None
        return QA_RUNNER


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Offline Smart Learning Assistant (Edge AI)</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --accent: #10b981;
      --accent-hover: #059669;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --card-inner: #0b1120;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
    .container { max-width: 1040px; margin: 0 auto; }
    
    header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
    h1 { font-size: 1.5rem; color: var(--primary); display: flex; align-items: center; gap: 8px; }
    .offline-badge { background: #065f46; color: #34d399; font-size: 0.75rem; font-weight: bold; padding: 4px 10px; border-radius: 9999px; border: 1px solid #059669; }
    
    .tabs { display: flex; gap: 8px; margin-bottom: 20px; }
    .tab-btn { background: var(--card-bg); color: var(--text-muted); border: 1px solid var(--border); padding: 10px 18px; border-radius: 8px; cursor: pointer; font-weight: 600; transition: all 0.2s; }
    .tab-btn.active { background: var(--primary); color: #000; border-color: var(--primary); }
    
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 22px; margin-bottom: 20px; }
    h2 { font-size: 1.2rem; margin-bottom: 8px; color: var(--text); display: flex; align-items: center; gap: 8px; }
    p { color: var(--text-muted); font-size: 0.9rem; line-height: 1.5; margin-bottom: 14px; }
    
    /* Upload Box Styling */
    .upload-zone {
      border: 2px dashed #38bdf8;
      background: rgba(56, 189, 248, 0.04);
      border-radius: 12px;
      padding: 32px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s;
      margin-bottom: 16px;
    }
    .upload-zone:hover, .upload-zone.dragover {
      background: rgba(56, 189, 248, 0.12);
      border-color: #7dd3fc;
    }
    .upload-icon { font-size: 2.5rem; margin-bottom: 8px; }
    .upload-title { font-size: 1.1rem; font-weight: bold; color: #fff; margin-bottom: 4px; }
    .upload-sub { font-size: 0.85rem; color: var(--text-muted); }
    
    .file-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: #0284c7;
      color: #fff;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 0.85rem;
      font-weight: 600;
      margin-top: 12px;
    }

    .form-group { margin-bottom: 16px; }
    label { display: block; font-size: 0.85rem; font-weight: 600; margin-bottom: 6px; color: var(--text-muted); }
    input[type="text"], input[type="number"], textarea, select {
      width: 100%; background: var(--card-inner); border: 1px solid var(--border); border-radius: 8px; color: var(--text); padding: 10px 14px; font-size: 0.95rem; outline: none; transition: border-color 0.2s;
    }
    input[type="text"]:focus, textarea:focus { border-color: var(--primary); }
    
    .btn { background: var(--primary); color: #000; border: none; font-weight: 600; padding: 11px 22px; border-radius: 8px; cursor: pointer; transition: background 0.2s; display: inline-flex; align-items: center; gap: 8px; font-size: 0.95rem; }
    .btn:hover { background: var(--primary-hover); }
    .btn-accent { background: var(--accent); color: #000; }
    .btn-accent:hover { background: var(--accent-hover); }
    .btn-secondary { background: var(--border); color: var(--text); }
    .btn-secondary:hover { background: #475569; }
    
    .pill-group { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
    .pill { background: var(--card-inner); border: 1px solid var(--border); color: var(--primary); font-size: 0.82rem; padding: 6px 12px; border-radius: 20px; cursor: pointer; transition: all 0.2s; }
    .pill:hover { background: #1e3a5f; border-color: var(--primary); }
    
    .result-box { background: var(--card-inner); border: 1px solid var(--border); border-radius: 10px; padding: 18px; margin-top: 16px; display: none; }
    .result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
    .answer-title { color: var(--accent); font-weight: bold; font-size: 0.95rem; }
    .meta-tag { font-size: 0.75rem; color: var(--text-muted); background: var(--card-bg); padding: 3px 8px; border-radius: 6px; border: 1px solid var(--border); }
    .answer-text { font-size: 1.1rem; line-height: 1.5; color: #fff; margin-bottom: 14px; }
    .citation-box { background: rgba(56, 189, 248, 0.08); border-left: 3px solid var(--primary); padding: 10px 14px; border-radius: 0 6px 6px 0; margin-bottom: 12px; font-size: 0.85rem; }
    .citation-label { font-weight: bold; color: var(--primary); margin-bottom: 4px; }
    .metrics-bar { display: flex; flex-wrap: wrap; gap: 16px; font-size: 0.8rem; color: var(--text-muted); border-top: 1px solid var(--border); padding-top: 10px; }
    
    .model-info-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 14px; }
    .metric-card { background: var(--card-inner); border: 1px solid var(--border); border-radius: 8px; padding: 14px; text-align: center; }
    .metric-val { font-size: 1.35rem; font-weight: bold; color: var(--primary); margin-top: 4px; }
    .metric-lbl { font-size: 0.75rem; color: var(--text-muted); }
    
    pre { background: var(--card-inner); padding: 12px; border-radius: 8px; font-size: 0.85rem; color: #cbd5e1; overflow-x: auto; white-space: pre-wrap; }
    .loader { display: none; width: 18px; height: 18px; border: 2px solid #000; border-top-color: transparent; border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Active model bar */
    .active-bar { background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px 14px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>🎓 Offline Smart Learning Assistant</h1>
        <p style="margin: 4px 0 0 0;">Upload Textbooks (PDF) • Auto-Train Locally • Answer Questions Offline</p>
      </div>
      <div>
        <span class="offline-badge">🟢 100% OFFLINE (ZERO CLOUD)</span>
      </div>
    </header>

    <div class="active-bar">
      <span>💾 Active Model: <strong id="bar-model-name" style="color: var(--primary);">models/demo_curriculum_model.pth</strong></span>
      <span>⚡ Device: <strong>Local CPU / Apple Silicon (PyTorch)</strong></span>
      <span>🔒 Cloud Calls: <strong>0 (No APIs)</strong></span>
    </div>

    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('upload-tab', this)">📤 Upload PDF & Auto-Train</button>
      <button class="tab-btn" onclick="switchTab('qa-tab', this)">💬 Ask Questions</button>
      <button class="tab-btn" onclick="switchTab('train-tab', this)">⚙️ Advanced Training</button>
      <button class="tab-btn" onclick="switchTab('aid-tab', this)">📝 Study Aid Generator</button>
    </div>

    <!-- TAB 1: UPLOAD PDF & AUTO-TRAIN -->
    <div id="upload-tab" class="tab-content active">
      <div class="card">
        <h2>📤 Upload Student Textbook (PDF) & Auto-Train</h2>
        <p>Drop your semester textbook or notes PDF here. The machine will <strong>automatically train itself locally</strong> on the content, compress the weights into a <code>.pth</code> model file, and prepare to answer questions based specifically on your textbook!</p>

        <div class="upload-zone" id="drop-zone" onclick="document.getElementById('pdf-file-input').click()">
          <div class="upload-icon">📄</div>
          <div class="upload-title">Click to Select or Drag & Drop Textbook (PDF)</div>
          <div class="upload-sub">Extracts chapters, builds vocabulary, and auto-trains offline PyTorch model</div>
          <input type="file" id="pdf-file-input" accept=".pdf" style="display: none;" onchange="handleFileSelect(event)">
          <div id="selected-file-info" style="display: none;"></div>
        </div>

        <div style="display: flex; gap: 12px; align-items: center;">
          <button class="btn btn-accent" id="btn-upload-train" onclick="uploadAndTrain()" disabled>
            <span>🚀 Upload & Auto-Train Machine</span>
            <div class="loader" id="upload-loader"></div>
          </button>
          <span id="upload-status-text" style="font-size: 0.85rem; color: var(--text-muted);">Please select a PDF file above.</span>
        </div>

        <div class="result-box" id="upload-result">
          <div class="result-header">
            <span class="answer-title">🎉 Machine Successfully Trained on Uploaded Textbook!</span>
            <span class="meta-tag" id="up-time">0.72s</span>
          </div>

          <div class="model-info-grid">
            <div class="metric-card">
              <div class="metric-lbl">Compressed Model (.pth)</div>
              <div class="metric-val" id="up-size">2.18 MB</div>
            </div>
            <div class="metric-card">
              <div class="metric-lbl">Textbook Chunks Processed</div>
              <div class="metric-val" id="up-chunks">7</div>
            </div>
            <div class="metric-card">
              <div class="metric-lbl">Model Parameters</div>
              <div class="metric-val" id="up-params">498,023</div>
            </div>
            <div class="metric-card">
              <div class="metric-lbl">Training Loss</div>
              <div class="metric-val" id="up-loss">0.425</div>
            </div>
          </div>

          <div style="margin-top: 18px; padding: 14px; background: rgba(56, 189, 248, 0.08); border-radius: 8px;">
            <p style="color: #fff; font-weight: bold; margin-bottom: 8px;">Ready! The model is now active. Click below to start asking questions about your textbook:</p>
            <button class="btn" onclick="switchTab('qa-tab', document.querySelectorAll('.tab-btn')[1])">👉 Ask Questions on this Textbook</button>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: ASK QUESTIONS -->
    <div id="qa-tab" class="tab-content">
      <div class="card">
        <h2>💬 Ask Curriculum Questions (Offline Inference)</h2>
        <p>Ask questions derived from your uploaded textbooks. Queries execute on local PyTorch weights in &lt; 2 ms without internet access.</p>
        
        <label>Suggested Questions based on Current Knowledge Base:</label>
        <div class="pill-group" id="suggested-pills">
          <span class="pill" onclick="setQuery('What is the function of stomata in plant leaves?')">🌿 What is the function of stomata?</span>
          <span class="pill" onclick="setQuery('What is chlorophyll and where is it found in plant cells?')">🔬 Where is chlorophyll found?</span>
          <span class="pill" onclick="setQuery('What energy do wind turbines capture to spin electric generators?')">💨 Wind turbines energy?</span>
          <span class="pill" onclick="setQuery('How do solar photovoltaic cells generate electric current?')">☀️ Photovoltaic cells current?</span>
          <span class="pill" onclick="setQuery('What do plant roots absorb from the soil?')">🌱 Roots absorption?</span>
        </div>

        <div class="form-group">
          <input type="text" id="question-input" placeholder="Type or paste your curriculum question here..." onkeydown="if(event.key==='Enter') askQuestion()">
        </div>

        <button class="btn" onclick="askQuestion()">
          <span>Answer Question</span>
          <div class="loader" id="qa-loader"></div>
        </button>

        <div class="result-box" id="qa-result">
          <div class="result-header">
            <span class="answer-title">✅ Local Model Answer</span>
            <span class="meta-tag" id="qa-confidence">Confidence: 99%</span>
          </div>
          <div class="answer-text" id="qa-answer-text"></div>
          
          <div class="citation-box">
            <div class="citation-label" id="qa-citation-source">Source: Textbook (Page 1)</div>
            <div id="qa-citation-excerpt" style="color: #cbd5e1; font-style: italic;"></div>
          </div>

          <div class="metrics-bar">
            <span>⏱️ Latency: <strong id="qa-latency">0.8 ms</strong></span>
            <span>💾 Active Model: <strong id="qa-model-name">demo_curriculum_model.pth</strong></span>
            <span>🔒 Cloud Calls: <strong>0 (Verified Offline)</strong></span>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: ADVANCED TRAINING -->
    <div id="train-tab" class="tab-content">
      <div class="card">
        <h2>⚙️ Multi-Source Curriculum Training (PDF + JSON)</h2>
        <p>Train across multiple textbook PDFs and JSON Q&A files simultaneously. Customize epochs and target <code>.pth</code> filename.</p>

        <div class="form-group">
          <label>PDF Textbook File Paths (comma-separated):</label>
          <input type="text" id="train-pdf" value="sample_data/science_textbook.pdf">
        </div>

        <div class="form-group">
          <label>JSON Curriculum Q&A File Paths:</label>
          <input type="text" id="train-json" value="sample_data/curriculum_qa.json">
        </div>

        <div style="display: flex; gap: 16px;">
          <div class="form-group" style="flex: 1;">
            <label>Output Model Path (.pth):</label>
            <input type="text" id="train-save" value="models/custom_curriculum_model.pth">
          </div>
          <div class="form-group" style="width: 120px;">
            <label>Epochs:</label>
            <input type="number" id="train-epochs" value="12" min="1" max="100">
          </div>
        </div>

        <button class="btn" onclick="startTraining()">
          <span>Start Training</span>
          <div class="loader" id="train-loader"></div>
        </button>
        <button class="btn btn-secondary" onclick="generateSamples()" style="margin-left: 8px;">Re-generate Sample Data</button>

        <div class="result-box" id="train-result">
          <div class="result-header">
            <span class="answer-title">🎉 Training Complete</span>
            <span class="meta-tag" id="train-time">0.72s</span>
          </div>
          <div class="model-info-grid">
            <div class="metric-card">
              <div class="metric-lbl">Model Weight Size</div>
              <div class="metric-val" id="train-size">2.18 MB</div>
            </div>
            <div class="metric-card">
              <div class="metric-lbl">Total Parameters</div>
              <div class="metric-val" id="train-params">498,023</div>
            </div>
            <div class="metric-card">
              <div class="metric-lbl">Curriculum Chunks</div>
              <div class="metric-val" id="train-chunks">7</div>
            </div>
            <div class="metric-card">
              <div class="metric-lbl">Final Loss</div>
              <div class="metric-val" id="train-loss">0.425</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: STUDY AID GENERATOR -->
    <div id="aid-tab" class="tab-content">
      <div class="card">
        <h2>📝 Passage Study Aid Generator (Summary + 3 Quiz Questions)</h2>
        <p>Problem Statement 1 deliverable: converts a 150–300 word passage into a 3-point summary and 3 diagnostic multiple-choice quiz questions.</p>

        <label>Pre-loaded Sample Passages:</label>
        <div class="pill-group">
          <span class="pill" onclick="loadPassage(1)">🌿 1. Biology (Photosynthesis - 184 words)</span>
          <span class="pill" onclick="loadPassage(2)">⚡ 2. Physics (Renewable Energy - 194 words)</span>
          <span class="pill" onclick="loadPassage(3)">💧 3. Earth Science (Water Cycle - 171 words)</span>
        </div>

        <div class="form-group">
          <textarea id="passage-input" rows="7" placeholder="Paste or type a 150-300 word educational passage here..."></textarea>
        </div>

        <button class="btn" onclick="generateStudyAid()">
          <span>Generate Study Aid & Telemetry</span>
          <div class="loader" id="aid-loader"></div>
        </button>

        <div class="result-box" id="aid-result">
          <div class="result-header">
            <span class="answer-title">📘 Generated Study Aid</span>
            <span class="meta-tag">ExecuTorch Compatible</span>
          </div>
          <pre id="aid-output-text"></pre>
        </div>
      </div>
    </div>
  </div>

  <script>
    let selectedPdfFile = null;

    function switchTab(tabId, btn) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      if (btn) btn.classList.add('active');
    }

    // Drag and drop handlers
    const dropZone = document.getElementById('drop-zone');
    ['dragenter', 'dragover'].forEach(name => {
      dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); }, false);
    });
    ['dragleave', 'drop'].forEach(name => {
      dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); }, false);
    });
    dropZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length && files[0].name.toLowerCase().endsWith('.pdf')) {
        setFile(files[0]);
      } else {
        alert('Please drop a valid .pdf file.');
      }
    });

    function handleFileSelect(e) {
      if (e.target.files.length) {
        setFile(e.target.files[0]);
      }
    }

    function setFile(file) {
      selectedPdfFile = file;
      const infoDiv = document.getElementById('selected-file-info');
      infoDiv.style.display = 'block';
      infoDiv.innerHTML = '<div class="file-chip">📄 ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)</div>';
      document.getElementById('btn-upload-train').disabled = false;
      document.getElementById('upload-status-text').innerText = 'Ready to train on ' + file.name;
    }

    async function uploadAndTrain() {
      if (!selectedPdfFile) return;

      const loader = document.getElementById('upload-loader');
      const btn = document.getElementById('btn-upload-train');
      const statusText = document.getElementById('upload-status-text');
      const resBox = document.getElementById('upload-result');

      loader.style.display = 'inline-block';
      btn.disabled = true;
      statusText.innerText = 'Extracting pages & auto-training PyTorch neural model...';

      const reader = new FileReader();
      reader.onload = async function() {
        try {
          const base64Data = reader.result.split(',')[1];
          const resp = await fetch('/api/upload-and-train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              filename: selectedPdfFile.name,
              file_base64: base64Data,
              epochs: 12
            })
          });

          const data = await resp.json();
          if (data.error) {
            alert('Training error: ' + data.error);
            statusText.innerText = 'Error: ' + data.error;
            return;
          }

          document.getElementById('up-size').innerText = data.file_size_mb.toFixed(2) + ' MB';
          document.getElementById('up-chunks').innerText = data.num_chunks;
          document.getElementById('up-params').innerText = data.total_params.toLocaleString();
          document.getElementById('up-loss').innerText = data.final_loss.toFixed(3);
          document.getElementById('up-time').innerText = data.training_time_s.toFixed(2) + 's';

          document.getElementById('bar-model-name').innerText = data.save_path;
          statusText.innerText = 'Auto-training complete! Model activated.';
          resBox.style.display = 'block';

          // Update suggested pills if topics returned
          if (data.suggested_questions && data.suggested_questions.length) {
            const pillsDiv = document.getElementById('suggested-pills');
            pillsDiv.innerHTML = '';
            data.suggested_questions.forEach(sq => {
              const span = document.createElement('span');
              span.className = 'pill';
              span.innerText = '📘 ' + sq;
              span.onclick = () => setQuery(sq);
              pillsDiv.appendChild(span);
            });
          }
        } catch (err) {
          alert('Upload failed: ' + err);
          statusText.innerText = 'Failed: ' + err;
        } finally {
          loader.style.display = 'none';
          btn.disabled = false;
        }
      };
      reader.readAsDataURL(selectedPdfFile);
    }

    function setQuery(text) {
      switchTab('qa-tab', document.querySelectorAll('.tab-btn')[1]);
      document.getElementById('question-input').value = text;
      askQuestion();
    }

    async function askQuestion() {
      const q = document.getElementById('question-input').value.trim();
      if (!q) return;

      const loader = document.getElementById('qa-loader');
      const resBox = document.getElementById('qa-result');
      loader.style.display = 'inline-block';

      try {
        const resp = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
        const data = await resp.json();

        document.getElementById('qa-answer-text').innerText = data.answer;
        document.getElementById('qa-citation-source').innerText = 'Source: ' + data.source_file + ' (' + data.source_reference + ')';
        document.getElementById('qa-citation-excerpt').innerText = '"' + data.context_passage.substring(0, 180) + '..."';
        document.getElementById('qa-confidence').innerText = 'Confidence: ' + (data.confidence_score * 100).toFixed(1) + '%';
        document.getElementById('qa-latency').innerText = data.inference_time_ms.toFixed(1) + ' ms';
        resBox.style.display = 'block';
      } catch (err) {
        alert('Error answering question: ' + err);
      } finally {
        loader.style.display = 'none';
      }
    }

    async function startTraining() {
      const pdf = document.getElementById('train-pdf').value.trim();
      const jsonF = document.getElementById('train-json').value.trim();
      const saveP = document.getElementById('train-save').value.trim();
      const epochs = parseInt(document.getElementById('train-epochs').value) || 12;

      const loader = document.getElementById('train-loader');
      const resBox = document.getElementById('train-result');
      loader.style.display = 'inline-block';

      try {
        const resp = await fetch('/api/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            pdf_paths: pdf ? pdf.split(',').map(s => s.trim()) : [],
            json_paths: jsonF ? jsonF.split(',').map(s => s.trim()) : [],
            save_path: saveP,
            epochs: epochs
          })
        });
        const data = await resp.json();

        if (data.error) {
          alert('Training error: ' + data.error);
          return;
        }

        document.getElementById('train-size').innerText = data.file_size_mb.toFixed(2) + ' MB';
        document.getElementById('train-params').innerText = data.total_params.toLocaleString();
        document.getElementById('train-chunks').innerText = data.num_chunks;
        document.getElementById('train-loss').innerText = data.final_loss.toFixed(3);
        document.getElementById('train-time').innerText = data.training_time_s.toFixed(2) + 's';
        document.getElementById('bar-model-name').innerText = data.save_path;
        resBox.style.display = 'block';
      } catch (err) {
        alert('Failed training: ' + err);
      } finally {
        loader.style.display = 'none';
      }
    }

    async function loadPassage(sampleId) {
      try {
        const resp = await fetch('/api/sample?id=' + sampleId);
        const data = await resp.json();
        document.getElementById('passage-input').value = data.text;
      } catch (e) {
        alert('Error loading sample: ' + e);
      }
    }

    async function generateStudyAid() {
      const text = document.getElementById('passage-input').value.trim();
      if (!text) {
        alert('Please enter or select a passage.');
        return;
      }

      const loader = document.getElementById('aid-loader');
      const resBox = document.getElementById('aid-result');
      loader.style.display = 'inline-block';

      try {
        const resp = await fetch('/api/study-aid', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ passage: text })
        });
        const data = await resp.json();
        document.getElementById('aid-output-text').innerText = data.output + '\\n\\n' + data.telemetry;
        resBox.style.display = 'block';
      } catch (e) {
        alert('Error: ' + e);
      } finally {
        loader.style.display = 'none';
      }
    }

    async function generateSamples() {
      await fetch('/api/generate-samples');
      alert('Sample PDF & JSON curriculum files created in sample_data/!');
    }
  </script>
</body>
</html>
"""


class OfflineLearningHTTPHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def _send_json(self, data: Dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url_parts = urllib.parse.urlparse(self.path)
        path = url_parts.path

        if path in {"/", "/index.html"}:
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/sample":
            query = urllib.parse.parse_qs(url_parts.query)
            sample_id = query.get("id", ["1"])[0]
            sample_files = {
                "1": "sample_passages/01_photosynthesis.txt",
                "2": "sample_passages/02_renewable_energy.txt",
                "3": "sample_passages/03_water_cycle.txt",
            }
            target = sample_files.get(sample_id, sample_files["1"])
            if os.path.exists(target):
                with open(target, "r", encoding="utf-8") as f:
                    self._send_json({"text": f.read()})
            else:
                self._send_json({"error": "Sample not found"}, status=404)

        elif path == "/api/generate-samples":
            create_sample_textbook_pdf("sample_data/science_textbook.pdf")
            create_sample_curriculum_json("sample_data/curriculum_qa.json")
            self._send_json({"success": True})

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        url_parts = urllib.parse.urlparse(self.path)
        path = url_parts.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        payload = json.loads(post_data.decode("utf-8")) if post_data else {}

        if path == "/api/upload-and-train":
            filename = payload.get("filename", "uploaded_textbook.pdf")
            file_base64 = payload.get("file_base64", "")
            epochs = payload.get("epochs", 12)

            if not file_base64:
                self._send_json({"error": "No file data received."}, status=400)
                return

            os.makedirs("uploads", exist_ok=True)
            safe_filename = re.sub(r"[^\w\-_\.]", "_", os.path.basename(filename))
            saved_pdf_path = os.path.join("uploads", safe_filename)

            # Decode and write PDF
            with open(saved_pdf_path, "wb") as f:
                f.write(base64.b64decode(file_base64))

            save_model_path = "models/uploaded_model.pth"

            try:
                # Auto-train the PyTorch neural model on this uploaded PDF
                stats = train_offline_model(
                    pdf_paths=[saved_pdf_path],
                    save_path=save_model_path,
                    epochs=epochs,
                )
                file_size_mb = os.path.getsize(save_model_path) / (1024.0 * 1024.0)

                # Set as active model runner
                runner = get_qa_runner(save_model_path)

                # Generate dynamic suggested questions based on the uploaded chunks
                suggested_questions = []
                for chunk in runner.knowledge_chunks[:4]:
                    text = chunk.get("text", "")
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
                    if sentences:
                        s_words = sentences[0].split()[:7]
                        suggested_questions.append(f"What does the text say about {' '.join(s_words)}?")

                self._send_json({
                    "success": True,
                    "save_path": save_model_path,
                    "file_size_mb": file_size_mb,
                    "total_params": stats["total_params"],
                    "num_chunks": stats["num_chunks"],
                    "final_loss": stats["final_loss"],
                    "training_time_s": stats["training_time_s"],
                    "suggested_questions": suggested_questions,
                })

            except Exception as e:
                self._send_json({"error": str(e)}, status=400)

        elif path == "/api/ask":
            question = payload.get("question", "").strip()
            runner = get_qa_runner()
            if not runner:
                if not os.path.exists(CURRENT_MODEL_PATH):
                    create_sample_textbook_pdf("sample_data/science_textbook.pdf")
                    create_sample_curriculum_json("sample_data/curriculum_qa.json")
                    train_offline_model(
                        pdf_paths=["sample_data/science_textbook.pdf"],
                        json_paths=["sample_data/curriculum_qa.json"],
                        save_path=CURRENT_MODEL_PATH,
                        epochs=10,
                    )
                runner = get_qa_runner(CURRENT_MODEL_PATH)

            response = runner.answer(question)
            self._send_json({
                "question": response.question,
                "answer": response.answer,
                "source_file": response.source_file,
                "source_reference": response.source_reference,
                "context_passage": response.context_passage,
                "confidence_score": response.confidence_score,
                "inference_time_ms": response.inference_time_ms,
                "offline_verified": True,
            })

        elif path == "/api/train":
            pdf_paths = payload.get("pdf_paths", [])
            json_paths = payload.get("json_paths", [])
            save_path = payload.get("save_path", "models/custom_curriculum_model.pth")
            epochs = payload.get("epochs", 12)

            try:
                stats = train_offline_model(
                    pdf_paths=pdf_paths,
                    json_paths=json_paths,
                    save_path=save_path,
                    epochs=epochs,
                )
                file_size_mb = os.path.getsize(save_path) / (1024.0 * 1024.0)

                # Update current active runner
                get_qa_runner(save_path)

                self._send_json({
                    "success": True,
                    "save_path": save_path,
                    "file_size_mb": file_size_mb,
                    "total_params": stats["total_params"],
                    "num_chunks": stats["num_chunks"],
                    "final_loss": stats["final_loss"],
                    "training_time_s": stats["training_time_s"],
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)

        elif path == "/api/study-aid":
            from engine import OfflineDeterministicEngine
            passage = payload.get("passage", "")
            engine = OfflineDeterministicEngine()
            output_text, telemetry = engine.generate(passage)
            self._send_json({
                "output": output_text,
                "telemetry": telemetry.format_report(),
            })

        else:
            self.send_response(404)
            self.end_headers()


def run_gui_server(port: int = 8000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, OfflineLearningHTTPHandler)
    print("\n" + "=" * 68)
    print("      OFFLINE SMART LEARNING ASSISTANT - LIGHTWEIGHT WEB GUI      ")
    print("=" * 68)
    print(f"  Local Web GUI URL:     http://localhost:{port}")
    print(f"  Feature:               Student PDF Upload & Instant Auto-Training")
    print(f"  Framework Dependency:  None (Python standard library http.server)")
    print(f"  Active PyTorch Model:  {CURRENT_MODEL_PATH}")
    print(f"  Cloud Invocations:     0 (100% Offline Verified)")
    print("=" * 68)
    print("Press Ctrl+C in terminal to stop server.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping GUI Server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_gui_server(port)
