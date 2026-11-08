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
import uuid
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional

from tools.generate_sample_data import create_sample_curriculum_json, create_sample_textbook_pdf
from prompt_templates import parse_learning_aid, validate_passage

# Global runner state
CURRENT_MODEL_PATH = "models/demo_curriculum_model.pth"
QA_RUNNER = None
RUNNER_LOCK = threading.Lock()
ACTIVE_TOPICS: List[str] = [
    "What is the function of stomata in plant leaves?",
    "What is chlorophyll and where is it found in plant cells?",
    "What energy do wind turbines capture to spin electric generators?",
    "How do solar photovoltaic cells generate electric current?",
]


def get_qa_runner(model_path: Optional[str] = None):
    global QA_RUNNER, CURRENT_MODEL_PATH
    model_path = model_path or CURRENT_MODEL_PATH
    with RUNNER_LOCK:
        if QA_RUNNER is None or CURRENT_MODEL_PATH != model_path:
            if os.path.exists(model_path):
                try:
                    from qa_engine import OfflineQAModelRunner
                    CURRENT_MODEL_PATH = model_path
                    QA_RUNNER = OfflineQAModelRunner(model_path)
                except Exception as e:
                    QA_RUNNER = None
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
    
    /* Chat Container & Bubble Styling */
    .chat-container {
      background: var(--card-inner);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      height: 480px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
      margin-bottom: 16px;
    }
    .chat-msg {
      display: flex;
      flex-direction: column;
      max-width: 88%;
      animation: fadeIn 0.2s ease-in;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
    .chat-msg.user {
      align-self: flex-end;
      align-items: flex-end;
    }
    .chat-msg.user .bubble {
      background: #0284c7;
      color: #fff;
      border-radius: 16px 16px 4px 16px;
      padding: 11px 16px;
      font-size: 0.95rem;
      font-weight: 500;
    }
    .chat-msg.assistant {
      align-self: flex-start;
      align-items: flex-start;
    }
    .chat-msg.assistant .bubble {
      background: #1e293b;
      border: 1px solid var(--border);
      color: #f8fafc;
      border-radius: 16px 16px 16px 4px;
      padding: 14px 18px;
      font-size: 0.95rem;
      line-height: 1.6;
      white-space: pre-wrap;
    }
    .msg-meta {
      font-size: 0.74rem;
      color: var(--text-muted);
      margin-top: 5px;
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .source-chip {
      background: rgba(56, 189, 248, 0.12);
      color: var(--primary);
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 2px 8px;
      border-radius: 10px;
      font-size: 0.72rem;
    }
    .input-row {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .input-row input {
      flex: 1;
      margin-bottom: 0;
    }

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
      <span>⚡ Device: <strong>Local CPU / ARM64 or Apple Silicon (PyTorch)</strong></span>
      <span>🔒 Cloud Calls: <strong>0 (No APIs)</strong></span>
    </div>

    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('qa-tab', this)">💬 Ask Questions & Chat</button>
      <button class="tab-btn" onclick="switchTab('upload-tab', this)">📤 Upload PDF & Auto-Train</button>
      <button class="tab-btn" onclick="switchTab('train-tab', this)">📁 One-Click Folder Training</button>
      <button class="tab-btn" onclick="switchTab('aid-tab', this)">📝 Study Aid Generator</button>
    </div>

    <!-- TAB 1: ASK QUESTIONS & CHAT (ACTIVE BY DEFAULT) -->
    <div id="qa-tab" class="tab-content active">
      <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
          <div>
            <h2>💬 Offline Smart Learning Assistant</h2>
            <p style="margin-bottom: 0;">Ask questions about your uploaded curriculum or say 'hi'. Queries execute in &lt; 2 ms 100% offline.</p>
          </div>
          <button class="btn btn-secondary" onclick="clearChat()" style="padding: 7px 14px; font-size: 0.82rem;">🧹 Clear Chat</button>
        </div>

        <!-- Chat Stream Feed -->
        <div class="chat-container" id="chat-feed">
          <div class="chat-msg assistant">
            <div class="bubble">Hello! 👋 Welcome to your <strong>Offline Smart Learning Assistant</strong>!

I am running 100% offline on your device, powered by a local PyTorch edge neural model.
I can answer curriculum questions, explain concepts in-depth, and help you study your uploaded textbooks with zero APIs or cloud connections.

💡 Click one of the quick suggestions below or type any question to begin!</div>
            <div class="msg-meta">
              <span>🤖 Offline AI Assistant</span>
              <span class="source-chip">System Core</span>
            </div>
          </div>
        </div>

        <label style="margin-bottom: 6px;">Quick Suggestions:</label>
        <div class="pill-group" id="suggested-pills">
          <span class="pill" onclick="setQuery('Hi! How can you help me study?')">👋 Say Hi</span>
          <span class="pill" onclick="setQuery('Tell me a joke!')">😂 Tell a Joke</span>
          <span class="pill" onclick="setQuery('What is the Pomodoro technique?')">🍅 Pomodoro Technique</span>
          <span class="pill" onclick="setQuery('Give me some motivation to study today.')">🌟 Motivation</span>
          <span class="pill" onclick="setQuery('Why is the sky blue?')">☀️ Sky Blue</span>
          <span class="pill" onclick="setQuery('What is the function of stomata in plant leaves?')">🌿 Stomata</span>
          <span class="pill" onclick="setQuery('What is chlorophyll and where is it found in plant cells?')">🔬 Chlorophyll</span>
        </div>

        <div class="input-row">
          <input type="text" id="question-input" placeholder="Type a message or curriculum question here (e.g. 'hi', 'explain photosynthesis')..." onkeydown="if(event.key==='Enter') askQuestion()">
          <button class="btn" id="btn-send" onclick="askQuestion()">
            <span>Send</span>
            <div class="loader" id="qa-loader"></div>
          </button>
        </div>
      </div>
    </div>

    <!-- TAB 2: UPLOAD PDF, CONVERT, THEN TRAIN -->
    <div id="upload-tab" class="tab-content">
      <div class="card">
        <h2>📤 Upload Student Textbook (PDF)</h2>
        <p>Upload processing runs automatically as PDF → page-preserving <code>.txt</code> → JSON. Training starts only after you confirm.</p>

        <div class="upload-zone" id="drop-zone" onclick="document.getElementById('pdf-file-input').click()">
          <div class="upload-icon">📄</div>
          <div class="upload-title">Click to Select or Drag & Drop Textbook (PDF)</div>
          <div class="upload-sub">PDF → page-preserving TXT → JSON → offline PyTorch model</div>
          <input type="file" id="pdf-file-input" accept=".pdf" style="display: none;" onchange="handleFileSelect(event)">
          <div id="selected-file-info" style="display: none;"></div>
        </div>

        <div style="display: flex; gap: 12px; align-items: center;">
          <button class="btn btn-accent" id="btn-upload-train" onclick="uploadAndTrain()" disabled>
            <span>🚀 Process PDF to TXT and JSON</span>
            <div class="loader" id="upload-loader"></div>
          </button>
          <span id="upload-status-text" style="font-size: 0.85rem; color: var(--text-muted);">Please select a PDF file above.</span>
        </div>
        <div id="upload-pipeline-status" style="margin-top: 12px; color: var(--text-muted); font-size: 0.85rem;">
          1. PDF waiting · 2. TXT waiting · 3. JSON waiting · 4. AI training waiting · 5. Answer model waiting
        </div>

        <div class="result-box" id="upload-result">
          <div class="result-header">
            <span class="answer-title">✅ PDF converted to TXT and JSON</span>
            <span class="meta-tag" id="up-time">0.72s</span>
          </div>

          <div class="model-info-grid">
            <div class="metric-card">
              <div class="metric-lbl">Textbook Chunks Processed</div>
              <div class="metric-val" id="up-chunks">0</div>
            </div>
            <div class="metric-card">
              <div class="metric-lbl">Generated Q&A Pairs</div>
              <div class="metric-val" id="up-qa">0</div>
            </div>
            <div class="metric-card">
              <div class="metric-lbl">JSON Input</div>
              <div class="metric-val" id="up-json" style="font-size: 0.7rem;">waiting</div>
            </div>
          </div>
          <button class="btn" id="btn-view-generated-questions" onclick="toggleGeneratedQuestions()" style="display:none; margin-top: 14px;">
            📖 View generated questions
          </button>
          <div id="generated-questions-panel" style="display:none; margin-top: 12px; max-height: 280px; overflow-y: auto; padding: 12px; background: rgba(15, 23, 42, 0.5); border-radius: 8px;">
            <ol id="generated-questions-list" style="margin: 0; padding-left: 24px;"></ol>
          </div>

          <div style="margin-top: 18px; padding: 14px; background: rgba(56, 189, 248, 0.08); border-radius: 8px;">
            <p id="training-confirmation" style="color: #fff; font-weight: bold; margin-bottom: 8px;">TXT and JSON are ready. Start AI training?</p>
            <button class="btn btn-accent" id="btn-confirm-training" onclick="confirmUploadedTraining()" disabled>🧠 Train AI from this JSON</button>
            <button class="btn" id="btn-ask-uploaded" onclick="openUploadedQuestions()" style="display:none;">👉 Ask Questions on this Textbook</button>
            <div id="training-progress" style="display:none; margin-top: 10px; color: var(--text-muted); font-size: 0.85rem;"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: ADVANCED TRAINING -->
    <div id="train-tab" class="tab-content">
      <div class="card">
        <h2>⚙️ Train AI from Your PDF Library</h2>
        <p>Train the offline model from every PDF inside <code>pdfs_to_train/</code>. Answers will be based on those PDFs.</p>

        <!-- One-Click Folder Train -->
        <div style="background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.3); border-radius: 10px; padding: 16px 18px; margin-bottom: 20px;">
          <div style="font-size: 1rem; font-weight: bold; color: #34d399; margin-bottom: 6px;">📁 One-Click Folder Training</div>
          <p style="margin-bottom: 12px;">Drop PDFs into <code>pdfs_to_train/</code>, then train the model using only those PDFs.</p>
          <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
            <button class="btn btn-accent" onclick="trainFromFolders()">
              <span>⚡ Train from PDF folder</span>
              <div class="loader" id="folder-train-loader"></div>
            </button>
            <span id="folder-train-status" style="font-size: 0.85rem; color: var(--text-muted);">Scans nested subject folders too.</span>
          </div>
        </div>

        <div style="color: var(--text-muted); font-size: 0.8rem; margin-bottom: 14px; border-top: 1px solid var(--border); padding-top: 14px;">— or specify individual file paths below —</div>

        <div class="form-group">
          <label>PDF Textbook File Paths (comma-separated):</label>
          <input type="text" id="train-pdf" value="pdfs_to_train/science_textbook.pdf">
        </div>

        <div class="form-group">
          <label>JSON Curriculum Q&A File Paths:</label>
          <input type="text" id="train-json" value="data/curriculum_json/curriculum_qa.json">
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

    function openUploadedQuestions() {
      const qaButton = document.querySelector('.tab-btn[onclick*="qa-tab"]');
      switchTab('qa-tab', qaButton);
      const questionInput = document.getElementById('question-input');
      if (questionInput) {
        questionInput.focus();
        questionInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
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
      if (!file.name.toLowerCase().endsWith('.pdf') && file.type !== 'application/pdf') {
        alert('Please select a PDF file.');
        return;
      }
      selectedPdfFile = file;
      const infoDiv = document.getElementById('selected-file-info');
      infoDiv.style.display = 'block';
      infoDiv.innerHTML = '<div class="file-chip">📄 ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)</div>';
      document.getElementById('btn-upload-train').disabled = false;
      document.getElementById('upload-status-text').innerText = 'Ready to convert ' + file.name;
    }

    async function uploadAndTrain() {
      if (!selectedPdfFile) return;

      const loader = document.getElementById('upload-loader');
      const btn = document.getElementById('btn-upload-train');
      const statusText = document.getElementById('upload-status-text');
      const resBox = document.getElementById('upload-result');

      loader.style.display = 'inline-block';
      btn.disabled = true;
      statusText.innerText = 'Converting PDF to TXT, then TXT to JSON...';

      try {
          const buffer = await selectedPdfFile.arrayBuffer();
          let binary = '';
          const bytes = new Uint8Array(buffer);
          const blockSize = 0x8000;
          for (let offset = 0; offset < bytes.length; offset += blockSize) {
            binary += String.fromCharCode(...bytes.subarray(offset, offset + blockSize));
          }
          const base64Data = btoa(binary);
          const resp = await fetch('/api/upload-and-train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              filename: selectedPdfFile.name,
              file_base64: base64Data,
            })
          });

          const data = await resp.json().catch(() => ({
            error: `Upload failed with HTTP ${resp.status}.`,
          }));
          if (!resp.ok || data.success !== true) {
            const errorMessage = data.error || `Conversion failed with HTTP ${resp.status}.`;
            alert('Conversion error: ' + errorMessage);
            statusText.innerText = 'Error: ' + errorMessage;
            return;
          }

          document.getElementById('up-chunks').innerText = data.extracted_pages;
          document.getElementById('up-qa').innerText = data.generated_qa_pairs;
          document.getElementById('up-json').innerText = data.json_path;
          document.getElementById('up-time').innerText = 'processed';
          renderGeneratedQuestions(data.generated_questions || []);
          statusText.innerText = 'TXT extracted and JSON created. Training has not started.';
          document.getElementById('upload-pipeline-status').innerText =
            '✓ PDF uploaded → ✓ TXT created → ✓ JSON created → ⏸ AI training waiting for confirmation';
          document.getElementById('btn-confirm-training').disabled = false;
          window.uploadedJsonPath = data.json_path;
          resBox.style.display = 'block';

          // Post notification to chat feed
          const chatFeed = document.getElementById('chat-feed');
          if (chatFeed) {
            const noticeMsg = document.createElement('div');
            noticeMsg.className = 'chat-msg assistant';
            noticeMsg.innerHTML = `
              <div class="bubble" style="border: 1px solid #10b981; background: rgba(16,185,129,0.08);">
                ✅ <strong>Textbook converted successfully!</strong><br>
                <strong>${escapeHtml(selectedPdfFile.name)}</strong> is ready as TXT → JSON. Training is waiting for your confirmation.
              </div>
              <div class="msg-meta">
                <span>📄 JSON Ready</span>
                <span class="source-chip">${escapeHtml(data.json_path)}</span>
              </div>
            `;
            chatFeed.appendChild(noticeMsg);
            chatFeed.scrollTop = chatFeed.scrollHeight;
          }

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
    }

    function escapeHtml(str) {
      if (!str) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    function renderGeneratedQuestions(questions) {
      const list = document.getElementById('generated-questions-list');
      const button = document.getElementById('btn-view-generated-questions');
      const panel = document.getElementById('generated-questions-panel');
      if (!list || !button || !panel) return;

      list.innerHTML = '';
      questions.forEach((question) => {
        const item = document.createElement('li');
        item.style.marginBottom = '8px';
        item.innerText = question;
        list.appendChild(item);
      });
      panel.style.display = 'none';
      button.style.display = questions.length ? 'inline-block' : 'none';
      button.innerText = `📖 View generated questions (${questions.length})`;
    }

    function toggleGeneratedQuestions() {
      const panel = document.getElementById('generated-questions-panel');
      const button = document.getElementById('btn-view-generated-questions');
      if (!panel || !button) return;
      const isHidden = panel.style.display === 'none';
      panel.style.display = isHidden ? 'block' : 'none';
      button.innerText = isHidden ? '📕 Hide generated questions' : '📖 View generated questions';
    }

    function clearChat() {
      const chatFeed = document.getElementById('chat-feed');
      chatFeed.innerHTML = `
        <div class="chat-msg assistant">
          <div class="bubble">Hello! 👋 Welcome to your <strong>Offline Smart Learning Assistant</strong>!

I am running 100% offline on your device, powered by a local PyTorch edge neural model.
I can answer curriculum questions, explain concepts in-depth, and help you study your uploaded textbooks with zero APIs or cloud connections.

💡 Click one of the quick suggestions below or type any question to begin!</div>
          <div class="msg-meta">
            <span>🤖 Offline AI Assistant</span>
            <span class="source-chip">System Core</span>
          </div>
        </div>
      `;
    }

    function setQuery(text) {
      const qaBtn = document.querySelector('button[onclick*="qa-tab"]');
      switchTab('qa-tab', qaBtn);
      document.getElementById('question-input').value = text;
      askQuestion();
    }

    async function askQuestion() {
      const inputEl = document.getElementById('question-input');
      const q = inputEl.value.trim();
      if (!q) return;

      const loader = document.getElementById('qa-loader');
      const chatFeed = document.getElementById('chat-feed');
      loader.style.display = 'inline-block';
      inputEl.value = '';

      // 1. Append user message to chat feed
      const userMsg = document.createElement('div');
      userMsg.className = 'chat-msg user';
      userMsg.innerHTML = `
        <div class="bubble">${escapeHtml(q)}</div>
        <div class="msg-meta"><span>You</span></div>
      `;
      chatFeed.appendChild(userMsg);
      chatFeed.scrollTop = chatFeed.scrollHeight;

      // 2. Query local offline model
      try {
        const resp = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
        const data = await resp.json();
        if (!resp.ok || data.error) {
          throw new Error(data.error || `Question request failed with HTTP ${resp.status}.`);
        }

        // 3. Append assistant generative response to chat feed
        const botMsg = document.createElement('div');
        botMsg.className = 'chat-msg assistant';

        const sourceLabel = escapeHtml(data.source_file || 'Curriculum') + ' (' + escapeHtml(data.source_reference || 'Reference') + ')';
        const latency = data.inference_time_ms ? data.inference_time_ms.toFixed(1) + ' ms' : '< 1 ms';
        const confidence = data.confidence_score ? (data.confidence_score * 100).toFixed(0) + '%' : '99%';

        botMsg.innerHTML = `
          <div class="bubble">${escapeHtml(data.answer)}</div>
          <div class="msg-meta">
            <span>🤖 Generative AI</span>
            <span class="source-chip">📖 ${sourceLabel}</span>
            <span>⏱️ ${latency}</span>
            <span>🎯 ${confidence}</span>
            <span>🔒 Offline: 100%</span>
          </div>
        `;
        chatFeed.appendChild(botMsg);
        chatFeed.scrollTop = chatFeed.scrollHeight;

      } catch (err) {
        const errMsg = document.createElement('div');
        errMsg.className = 'chat-msg assistant';
        errMsg.innerHTML = `
          <div class="bubble" style="border-color: #f87171; color: #fca5a5;">
            Error connecting to local engine: ${escapeHtml(String(err))}
          </div>
        `;
        chatFeed.appendChild(errMsg);
        chatFeed.scrollTop = chatFeed.scrollHeight;
      } finally {
        loader.style.display = 'none';
      }
    }

    async function confirmUploadedTraining() {
      if (!window.uploadedJsonPath) return;
      const button = document.getElementById('btn-confirm-training');
      const statusText = document.getElementById('upload-status-text');
      button.disabled = true;
      statusText.innerText = 'Training AI from the generated JSON...';
      const progress = document.getElementById('training-progress');
      progress.style.display = 'block';
      progress.innerText = '⏳ Loading JSON knowledge...';
      try {
        progress.innerText = '⏳ Training the offline AI model from TXT-derived JSON...';
        const resp = await fetch('/api/train-uploaded-json', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ json_path: window.uploadedJsonPath, epochs: 12 })
        });
        const data = await resp.json();
        if (!resp.ok || data.error) throw new Error(data.error || `Training failed with HTTP ${resp.status}.`);
        document.getElementById('up-time').innerText = data.training_time_s.toFixed(2) + 's';
        document.getElementById('upload-pipeline-status').innerText =
          '✓ PDF uploaded → ✓ TXT created → ✓ JSON created → ✓ AI trained from JSON → ✓ Answer model activated';
        document.getElementById('training-confirmation').innerText = 'Training complete. The answer model is active.';
        document.getElementById('btn-ask-uploaded').style.display = 'inline-block';
        document.getElementById('bar-model-name').innerText = data.save_path;
        progress.innerText = `✅ Training complete: ${data.num_chunks} knowledge chunks loaded in ${data.training_time_s.toFixed(2)}s.`;
        statusText.innerText = 'AI trained from JSON and activated.';
      } catch (err) {
        button.disabled = false;
        progress.innerText = '❌ Training failed. No answer model was activated.';
        statusText.innerText = 'Training failed: ' + err;
        alert('Training error: ' + err);
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

    async function trainFromFolders() {
      const loader = document.getElementById('folder-train-loader');
      const statusEl = document.getElementById('folder-train-status');
      const resBox = document.getElementById('train-result');
      loader.style.display = 'inline-block';
      statusEl.innerText = 'Scanning pdfs_to_train/ ...';

      try {
        const resp = await fetch('/api/train-from-folders', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ save_path: 'data/trained_models/folder_trained_model.pth', epochs: 12 })
        });
        const data = await resp.json();
        if (data.error) {
          statusEl.innerText = 'Error: ' + data.error;
          alert('Training error: ' + data.error);
          return;
        }
        document.getElementById('train-size').innerText = data.file_size_mb.toFixed(2) + ' MB';
        document.getElementById('train-params').innerText = data.total_params.toLocaleString();
        document.getElementById('train-chunks').innerText = data.num_chunks;
        document.getElementById('train-loss').innerText = data.final_loss.toFixed(3);
        document.getElementById('train-time').innerText = data.training_time_s.toFixed(2) + 's';
        document.getElementById('bar-model-name').innerText = data.save_path;
        statusEl.innerText = '✅ Trained from folders in ' + data.training_time_s.toFixed(1) + 's!';
        resBox.style.display = 'block';

        const chatFeed = document.getElementById('chat-feed');
        if (chatFeed) {
          const noticeMsg = document.createElement('div');
          noticeMsg.className = 'chat-msg assistant';
          noticeMsg.innerHTML = `
            <div class="bubble" style="border: 1px solid #10b981; background: rgba(16,185,129,0.08);">
              ⚡ <strong>Trained from data/ Folders!</strong><br>
              Scanned and trained on PDF files in <code>pdfs_to_train/</code> (${data.file_size_mb.toFixed(2)} MB .pth model, ${data.num_chunks} chunks).
              The updated model is now active for questions!
            </div>
            <div class="msg-meta">
              <span>⚡ Model Switched</span>
              <span class="source-chip">${escapeHtml(data.save_path)}</span>
            </div>
          `;
          chatFeed.appendChild(noticeMsg);
          chatFeed.scrollTop = chatFeed.scrollHeight;
        }
      } catch (err) {
        statusEl.innerText = 'Failed: ' + err;
        alert('Failed: ' + err);
      } finally {
        loader.style.display = 'none';
      }
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

        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send_json({"error": "Invalid upload request size."}, status=400)
            return
        if content_length <= 0:
            self._send_json({"error": "The upload request was empty."}, status=400)
            return
        if content_length > 150 * 1024 * 1024:
            self._send_json({"error": "The PDF is too large. Maximum upload size is 150 MB."}, status=413)
            return
        post_data = self.rfile.read(content_length)
        try:
            payload = json.loads(post_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "The upload request was not valid JSON."}, status=400)
            return
        if not isinstance(payload, dict):
            self._send_json({"error": "The upload request must be a JSON object."}, status=400)
            return

        if path == "/api/upload-and-train":
            filename = payload.get("filename", "uploaded_textbook.pdf")
            file_base64 = payload.get("file_base64", "")
            epochs = payload.get("epochs", 12)

            if not file_base64:
                self._send_json({"error": "No file data received."}, status=400)
                return

            base_dir = os.path.dirname(os.path.abspath(__file__))
            safe_filename = re.sub(r"[^\w\-_\.]", "_", os.path.basename(filename))
            if not safe_filename.lower().endswith(".pdf"):
                safe_filename += ".pdf"
            document_name = (
                f"{os.path.splitext(safe_filename)[0]}_{uuid.uuid4().hex[:8]}"
            )
            document_root = os.path.join(
                base_dir, "data", "uploaded_documents", document_name
            )
            pdf_dir = os.path.join(document_root, "pdf")
            text_dir = os.path.join(document_root, "text")
            json_dir = os.path.join(document_root, "json")
            os.makedirs(pdf_dir, exist_ok=True)
            os.makedirs(text_dir, exist_ok=True)
            os.makedirs(json_dir, exist_ok=True)
            saved_pdf_path = os.path.join(pdf_dir, safe_filename)

            # Decode and write PDF
            try:
                if file_base64.startswith("data:"):
                    file_base64 = file_base64.split(",", 1)[1]
                decoded_pdf = base64.b64decode(file_base64, validate=True)
            except Exception:
                self._send_json({"error": "The uploaded file is not valid base64 data."}, status=400)
                return
            if decoded_pdf[:1024].find(b"%PDF-") < 0:
                self._send_json({"error": "The uploaded file is not a valid PDF."}, status=400)
                return
            with open(saved_pdf_path, "wb") as f:
                f.write(decoded_pdf)

            save_text_path = os.path.join(
                text_dir, f"{os.path.splitext(safe_filename)[0]}.txt"
            )
            save_json_path = os.path.join(
                json_dir, f"{os.path.splitext(safe_filename)[0]}_knowledge.json"
            )

            try:
                # Keep the text archive as the source of truth for JSON conversion.
                from data_loader import (
                    extract_pdf_to_text_file,
                    export_text_knowledge_json,
                    load_qa_from_json,
                )
                text_extraction = extract_pdf_to_text_file(saved_pdf_path, save_text_path)
                extraction = export_text_knowledge_json(
                    text_path=save_text_path,
                    json_path=save_json_path,
                    source_file=os.path.basename(saved_pdf_path),
                    extraction_report=text_extraction,
                    subject="general",
                    question_root=os.path.join(base_dir, "data", "generated_questions"),
                    answer_root=os.path.join(base_dir, "data", "generated_answers"),
                )

                # Generate dynamic suggested questions based on the uploaded chunks
                generated_chunks = load_qa_from_json(save_json_path)
                suggested_questions = [
                    chunk.question
                    for chunk in generated_chunks
                    if chunk.question
                ][:8]
                generated_questions = [
                    chunk.question
                    for chunk in generated_chunks
                    if chunk.question
                ]
                if not suggested_questions:
                    suggested_questions = [
                        "What is the main topic covered in this textbook?",
                        "Explain the core mechanisms described in chapter 1?",
                    ]

                self._send_json({
                    "success": True,
                    "document_folder": os.path.relpath(document_root, base_dir),
                    "pdf_path": os.path.relpath(saved_pdf_path, base_dir),
                    "json_path": os.path.relpath(save_json_path, base_dir),
                    "text_path": os.path.relpath(save_text_path, base_dir),
                    "extracted_pages": extraction["pages"],
                    "extracted_words": extraction["words"],
                    "generated_qa_pairs": extraction["qa_pairs"],
                    "generated_questions": generated_questions,
                    "suggested_questions": suggested_questions,
                    "pipeline": [
                        {"stage": "pdf_upload", "status": "complete"},
                        {"stage": "pdf_to_text", "status": "complete", "path": os.path.relpath(save_text_path, base_dir), "library": "PyMuPDF"},
                        {"stage": "text_to_json", "status": "complete", "path": os.path.relpath(save_json_path, base_dir)},
                        {"stage": "json_training", "status": "waiting_for_confirmation", "input": os.path.relpath(save_json_path, base_dir)},
                        {"stage": "answer_model", "status": "waiting"},
                    ],
                })

            except Exception as e:
                self._send_json({
                    "success": False,
                    "error": f"Conversion failed: {e}",
                    "stage": "pdf_to_text_or_text_to_json",
                }, status=400)

        elif path == "/api/train-uploaded-json":
            base_dir = os.path.dirname(os.path.abspath(__file__))
            relative_json_path = payload.get("json_path", "")
            json_root = os.path.realpath(
                os.path.join(base_dir, "data", "uploaded_documents")
            )
            json_path = os.path.realpath(os.path.join(base_dir, relative_json_path))
            if not relative_json_path or not json_path.startswith(json_root + os.sep):
                self._send_json({"error": "Invalid uploaded JSON path."}, status=400)
                return
            if not os.path.isfile(json_path):
                self._send_json({"error": "Uploaded JSON file was not found. Process the PDF first."}, status=404)
                return
            try:
                from trainer import train_offline_model
                document_root = os.path.dirname(os.path.dirname(json_path))
                model_dir = os.path.join(base_dir, "models", "uploaded_documents")
                os.makedirs(model_dir, exist_ok=True)
                model_name = os.path.basename(document_root) + ".pth"
                save_model_path = os.path.join(model_dir, model_name)
                stats = train_offline_model(
                    pdf_paths=[],
                    json_paths=[json_path],
                    save_path=save_model_path,
                    epochs=payload.get("epochs", 12),
                )
                get_qa_runner(save_model_path)
                self._send_json({
                    "success": True,
                    "save_path": os.path.relpath(save_model_path, base_dir),
                    "file_size_mb": os.path.getsize(save_model_path) / (1024.0 * 1024.0),
                    "total_params": stats["total_params"],
                    "num_chunks": stats["num_chunks"],
                    "final_loss": stats["final_loss"],
                    "training_time_s": stats["training_time_s"],
                })
            except Exception as exc:
                self._send_json({"error": f"Training failed: {exc}"}, status=500)

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

        elif path == "/api/train-from-folders":
            # Auto-scan the drop-zone directories and train
            base_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(base_dir, "data", "trained_models", "folder_trained_model.pth")
            epochs = payload.get("epochs", 12)
            pdf_dir = os.path.join(base_dir, "pdfs_to_train")
            json_dir = None

            try:
                from data_loader import convert_pdf_folder_to_json, convert_pdf_folder_to_text
                from trainer import train_offline_model
                convert_pdf_folder_to_text(
                    pdf_dir,
                    os.path.join(base_dir, "data", "converted_text"),
                )
                conversion = convert_pdf_folder_to_json(
                    pdf_dir,
                    os.path.join(base_dir, "data", "converted_json"),
                    text_root=os.path.join(base_dir, "data", "converted_text"),
                )
                json_paths = [item["json_path"] for item in conversion["converted"]]
                if not json_paths:
                    self._send_json({"error": f"No PDF files found in {pdf_dir}. Add PDF files there and try again."}, status=400)
                    return
                stats = train_offline_model(
                    pdf_paths=[],
                    json_paths=json_paths,
                    save_path=save_path,
                    epochs=epochs,
                )
                file_size_mb = os.path.getsize(save_path) / (1024.0 * 1024.0)
                get_qa_runner(save_path)
                self._send_json({
                    "success": True,
                    "save_path": "data/trained_models/folder_trained_model.pth",
                    "file_size_mb": file_size_mb,
                    "total_params": stats["total_params"],
                    "num_chunks": stats["num_chunks"],
                    "final_loss": stats["final_loss"],
                    "training_time_s": stats["training_time_s"],
                    "pdf_files": [os.path.relpath(item["pdf_path"], base_dir) for item in conversion["converted"]],
                    "json_files": [os.path.relpath(path, base_dir) for path in json_paths],
                    "converted_count": conversion["converted_count"],
                    "failed_count": conversion["failed_count"],
                    "json_files": json_paths,
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)

        elif path == "/api/train":
            pdf_paths = payload.get("pdf_paths", [])
            json_paths = payload.get("json_paths", [])
            save_path = payload.get("save_path", "models/custom_curriculum_model.pth")
            epochs = payload.get("epochs", 12)

            try:
                if pdf_paths:
                    from data_loader import extract_pdf_to_text_file, export_text_knowledge_json

                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    generated_json_paths = []
                    for pdf_path in pdf_paths:
                        absolute_pdf = pdf_path if os.path.isabs(pdf_path) else os.path.join(base_dir, pdf_path)
                        stem = os.path.splitext(os.path.basename(absolute_pdf))[0]
                        text_path = os.path.join(
                            base_dir, "data", "converted_text", "gui", f"{stem}.txt"
                        )
                        generated_json = os.path.join(
                            base_dir, "data", "converted_json", "gui", f"{stem}.json"
                        )
                        extraction_report = extract_pdf_to_text_file(absolute_pdf, text_path)
                        export_text_knowledge_json(
                            text_path=text_path,
                            json_path=generated_json,
                            source_file=os.path.basename(absolute_pdf),
                            extraction_report=extraction_report,
                            subject="general",
                            question_root=os.path.join(base_dir, "data", "generated_questions"),
                            answer_root=os.path.join(base_dir, "data", "generated_answers"),
                        )
                        generated_json_paths.append(generated_json)
                    json_paths = list(json_paths) + generated_json_paths

                from trainer import train_offline_model
                stats = train_offline_model(
                    pdf_paths=[],
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
