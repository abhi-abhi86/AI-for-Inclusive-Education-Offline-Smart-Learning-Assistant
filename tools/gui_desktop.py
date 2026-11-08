"""
Native Desktop GUI for Offline Smart Learning Assistant (Tkinter).
Runs on macOS, Linux, and Raspberry Pi without any external GUI packages.
Provides input-to-output workflow for:
1. Asking curriculum questions with instant offline answers and source citations.
2. Training neural models on local PDF & JSON documents into compressed .pth checkpoints.
3. Generating 3-point summaries and 3 diagnostic quiz questions.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from prompt_templates import validate_passage
from qa_engine import OfflineQAModelRunner
from trainer import train_offline_model


def _convert_pdf_to_json_for_training(pdf_path: str, output_root: str) -> str:
    """Create the TXT and JSON archives used by the training model."""
    from data_loader import extract_pdf_to_text_file, export_text_knowledge_json

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    text_path = os.path.join(output_root, "converted_text", f"{stem}.txt")
    json_path = os.path.join(output_root, "converted_json", f"{stem}.json")
    extraction = extract_pdf_to_text_file(pdf_path, text_path)
    export_text_knowledge_json(
        text_path=text_path,
        json_path=json_path,
        source_file=os.path.basename(pdf_path),
        extraction_report=extraction,
        subject="general",
        question_root=os.path.join(output_root, "generated_questions"),
        answer_root=os.path.join(output_root, "generated_answers"),
    )
    return json_path


class OfflineLearningAssistantDesktopApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Offline Smart Learning Assistant (Edge AI)")
        self.root.geometry("820x680")
        self.root.minsize(720, 560)

        # Active Model
        self.default_model_path = "models/demo_curriculum_model.pth"
        self.qa_runner = None
        self._load_active_model()

        self._build_ui()

    def _load_active_model(self):
        if os.path.exists(self.default_model_path):
            try:
                self.qa_runner = OfflineQAModelRunner(self.default_model_path)
            except Exception as e:
                print(f"Warning loading model: {e}")
                self.qa_runner = None

    def _build_ui(self):
        # Header banner
        header_frame = tk.Frame(self.root, bg="#0f172a", padx=16, pady=12)
        header_frame.pack(fill="x")

        title_lbl = tk.Label(
            header_frame,
            text="🎓 Offline Smart Learning Assistant",
            font=("Helvetica", 16, "bold"),
            fg="#38bdf8",
            bg="#0f172a",
        )
        title_lbl.pack(side="left")

        badge_lbl = tk.Label(
            header_frame,
            text="🟢 100% OFFLINE (ZERO APIS)",
            font=("Helvetica", 9, "bold"),
            fg="#34d399",
            bg="#064e3b",
            padx=8,
            pady=3,
        )
        badge_lbl.pack(side="right")

        # Notebook tabs
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background="#1e293b")
        style.configure("TNotebook.Tab", padding=[16, 8], font=("Helvetica", 10, "bold"))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=10)

        # Tab 1: Offline Q&A
        self.qa_tab = tk.Frame(self.notebook, bg="#f8fafc", padx=16, pady=14)
        self.notebook.add(self.qa_tab, text="💬 Ask Questions")
        self._build_qa_tab()

        # Tab 2: Train Model
        self.train_tab = tk.Frame(self.notebook, bg="#f8fafc", padx=16, pady=14)
        self.notebook.add(self.train_tab, text="⚙️ Train on PDF / JSON")
        self._build_train_tab()

        # Tab 3: Study Aid Generator
        self.aid_tab = tk.Frame(self.notebook, bg="#f8fafc", padx=16, pady=14)
        self.notebook.add(self.aid_tab, text="📝 Study Aid Generator")
        self._build_aid_tab()

    # ------------------ TAB 1: QA INTERFACE ------------------
    def _build_qa_tab(self):
        # Input Section
        input_frame = tk.LabelFrame(self.qa_tab, text=" Input: Curriculum Question ", font=("Helvetica", 10, "bold"), padx=12, pady=8)
        input_frame.pack(fill="x", pady=6)

        pills_frame = tk.Frame(input_frame)
        pills_frame.pack(fill="x", pady=4)

        presets = [
            ("👋 Say Hi", "Hi! How can you help me study?"),
            ("😂 Tell Joke", "Tell me a joke!"),
            ("🍅 Pomodoro", "What is the Pomodoro technique?"),
            ("🌿 Stomata", "What is the function of stomata in plant leaves?"),
            ("🔬 Chlorophyll", "What is chlorophyll and where is it found in plant cells?"),
            ("☀️ Solar PV", "How do solar photovoltaic cells generate electric current?"),
        ]
        for label, q_text in presets:
            btn = tk.Button(
                pills_frame,
                text=label,
                font=("Helvetica", 8),
                bg="#e2e8f0",
                command=lambda qt=q_text: self._set_query(qt),
            )
            btn.pack(side="left", padx=4)

        entry_frame = tk.Frame(input_frame)
        entry_frame.pack(fill="x", pady=6)

        self.q_entry = tk.Entry(entry_frame, font=("Helvetica", 12))
        self.q_entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=4)
        self.q_entry.bind("<Return>", lambda e: self._on_ask())

        self.ask_btn = tk.Button(
            entry_frame,
            text="Ask Question",
            font=("Helvetica", 10, "bold"),
            bg="#0284c7",
            fg="white",
            padx=12,
            command=self._on_ask,
        )
        self.ask_btn.pack(side="right")

        self.clear_btn = tk.Button(
            entry_frame,
            text="🧹 Clear",
            font=("Helvetica", 9),
            bg="#e2e8f0",
            padx=8,
            command=self._on_clear_chat,
        )
        self.clear_btn.pack(side="right", padx=(0, 6))

        # Output Section
        output_frame = tk.LabelFrame(self.qa_tab, text=" Conversation & Generative AI Answers (100% Offline) ", font=("Helvetica", 10, "bold"), padx=12, pady=8)
        output_frame.pack(fill="both", expand=True, pady=6)

        self.ans_text = tk.Text(output_frame, height=12, font=("Helvetica", 11), wrap="word", bg="#f8fafc", padx=10, pady=10)
        self.ans_text.pack(fill="both", expand=True, pady=4)

        self.ans_text.tag_configure("user_q", font=("Helvetica", 11, "bold"), foreground="#0284c7")
        self.ans_text.tag_configure("ai_ans", font=("Helvetica", 11), foreground="#0f172a")
        self.ans_text.tag_configure("source_tag", font=("Helvetica", 9, "italic"), foreground="#059669")
        self.ans_text.tag_configure("divider", foreground="#94a3b8")

        self.ans_text.insert(
            tk.END,
            "🤖 Welcome to your Offline Smart Learning Assistant!\n"
            "I run 100% offline on your device, ready to explain concepts from your textbooks.\n"
            "Type a question above or click any preset pill to begin.\n\n",
            "ai_ans",
        )

        # Meta info
        meta_frame = tk.Frame(output_frame)
        meta_frame.pack(fill="x", pady=4)

        self.lbl_source = tk.Label(meta_frame, text="Source: None", font=("Helvetica", 9, "bold"), fg="#0369a1")
        self.lbl_source.pack(side="left")

        self.lbl_confidence = tk.Label(meta_frame, text="Confidence: --", font=("Helvetica", 9), fg="#059669")
        self.lbl_confidence.pack(side="left", padx=16)

        self.lbl_latency = tk.Label(meta_frame, text="Latency: -- ms", font=("Helvetica", 9), fg="#475569")
        self.lbl_latency.pack(side="right")

    def _on_clear_chat(self):
        self.ans_text.delete("1.0", tk.END)
        self.ans_text.insert(
            tk.END,
            "🤖 Conversation cleared.\nType a question above or click a preset pill to begin.\n\n",
            "ai_ans",
        )

    def _set_query(self, text):
        self.q_entry.delete(0, tk.END)
        self.q_entry.insert(0, text)
        self._on_ask()

    def _on_ask(self):
        query = self.q_entry.get().strip()
        if not query:
            return

        if not self.qa_runner:
            self._load_active_model()
            if not self.qa_runner:
                messagebox.showerror("Model Missing", "No model found. Please train a model in the 'Train' tab first.")
                return

        resp = self.qa_runner.answer(query)
        self.q_entry.delete(0, tk.END)

        # Append question and answer to conversation
        if self.ans_text.get("1.0", tk.END).strip():
            self.ans_text.insert(tk.END, "\n" + "—" * 60 + "\n\n", "divider")
        self.ans_text.insert(tk.END, f"👤 Student: {query}\n\n", "user_q")
        self.ans_text.insert(tk.END, f"🤖 Assistant:\n{resp.answer}\n\n", "ai_ans")
        self.ans_text.insert(tk.END, f"📖 Source: {resp.source_file} ({resp.source_reference})\n", "source_tag")
        self.ans_text.see(tk.END)

        self.lbl_source.config(text=f"Source: {resp.source_file} ({resp.source_reference})")
        self.lbl_confidence.config(text=f"Confidence: {resp.confidence_score * 100:.1f}%")
        self.lbl_latency.config(text=f"Latency: {resp.inference_time_ms:.1f} ms | Mode: Generative AI | Offline: 100%")

    # ------------------ TAB 2: TRAINING INTERFACE ------------------
    def _build_train_tab(self):
        # Quick Auto-Train Card
        auto_frame = tk.LabelFrame(self.train_tab, text=" ⚡ Student Quick Upload & Auto-Train ", font=("Helvetica", 10, "bold"), padx=12, pady=10, fg="#0284c7")
        auto_frame.pack(fill="x", pady=6)

        tk.Label(auto_frame, text="Select any semester textbook PDF. The machine will auto-train locally and activate itself for Q&A:").pack(anchor="w", pady=2)
        
        quick_btn_frame = tk.Frame(auto_frame)
        quick_btn_frame.pack(fill="x", pady=6)

        self.btn_auto_upload = tk.Button(
            quick_btn_frame,
            text="📂 Select PDF & Auto-Train Machine",
            font=("Helvetica", 10, "bold"),
            bg="#0284c7",
            fg="white",
            padx=14,
            pady=6,
            command=self._on_auto_upload_pdf,
        )
        self.btn_auto_upload.pack(side="left")

        self.lbl_auto_status = tk.Label(quick_btn_frame, text="Ready.", font=("Helvetica", 9), fg="#64748b")
        self.lbl_auto_status.pack(side="left", padx=12)

        # One-Click Folder Training
        folder_frame = tk.LabelFrame(self.train_tab, text=" ⚡ One-Click Folder Training ", font=("Helvetica", 10, "bold"), padx=12, pady=10, fg="#059669")
        folder_frame.pack(fill="x", pady=4)

        tk.Label(folder_frame, text="Drop PDFs → data/reference_pdfs/   |   Drop JSON → data/curriculum_json/   Then click:").pack(anchor="w", pady=2)

        folder_btn_row = tk.Frame(folder_frame)
        folder_btn_row.pack(fill="x", pady=4)

        self.btn_folder_train = tk.Button(
            folder_btn_row,
            text="⚡ Train from data/ Folders",
            font=("Helvetica", 10, "bold"),
            bg="#059669",
            fg="white",
            padx=14,
            pady=6,
            command=self._on_folder_train,
        )
        self.btn_folder_train.pack(side="left")

        self.lbl_folder_status = tk.Label(folder_btn_row, text="Scans data/reference_pdfs/ and data/curriculum_json/ automatically.", font=("Helvetica", 9), fg="#64748b")
        self.lbl_folder_status.pack(side="left", padx=12)

        # Advanced Manual Training Card
        input_frame = tk.LabelFrame(self.train_tab, text=" Advanced Multi-Source Training (PDF & JSON) ", font=("Helvetica", 10, "bold"), padx=12, pady=10)
        input_frame.pack(fill="x", pady=6)

        # PDF file selector
        tk.Label(input_frame, text="PDF Textbooks:").grid(row=0, column=0, sticky="w", pady=4)
        self.entry_pdf = tk.Entry(input_frame, font=("Helvetica", 10), width=48)
        self.entry_pdf.insert(0, "data/reference_pdfs/science_textbook.pdf")
        self.entry_pdf.grid(row=0, column=1, padx=6, pady=4, sticky="ew")
        tk.Button(input_frame, text="Browse...", command=self._browse_pdf).grid(row=0, column=2, padx=4)

        # JSON file selector
        tk.Label(input_frame, text="JSON Q&A / Curriculum:").grid(row=1, column=0, sticky="w", pady=4)
        self.entry_json = tk.Entry(input_frame, font=("Helvetica", 10), width=48)
        self.entry_json.insert(0, "data/curriculum_json/curriculum_qa.json")
        self.entry_json.grid(row=1, column=1, padx=6, pady=4, sticky="ew")
        tk.Button(input_frame, text="Browse...", command=self._browse_json).grid(row=1, column=2, padx=4)

        # Output save path & epochs
        tk.Label(input_frame, text="Save Model (.pth):").grid(row=2, column=0, sticky="w", pady=4)
        self.entry_save = tk.Entry(input_frame, font=("Helvetica", 10), width=48)
        self.entry_save.insert(0, "models/desktop_trained_model.pth")
        self.entry_save.grid(row=2, column=1, padx=6, pady=4, sticky="ew")

        epochs_frame = tk.Frame(input_frame)
        epochs_frame.grid(row=3, column=1, sticky="w", pady=6)
        tk.Label(epochs_frame, text="Epochs:").pack(side="left")
        self.spin_epochs = tk.Spinbox(epochs_frame, from_=1, to=100, width=5)
        self.spin_epochs.delete(0, "end")
        self.spin_epochs.insert(0, "12")
        self.spin_epochs.pack(side="left", padx=6)

        self.btn_train = tk.Button(
            input_frame,
            text="🚀 Start Local Training",
            font=("Helvetica", 10, "bold"),
            bg="#10b981",
            fg="white",
            padx=12,
            pady=4,
            command=self._on_train,
        )
        self.btn_train.grid(row=4, column=1, sticky="w", pady=8)

        # Output Section
        output_frame = tk.LabelFrame(self.train_tab, text=" Output: Training Telemetry & Model Stats ", font=("Helvetica", 10, "bold"), padx=12, pady=10)
        output_frame.pack(fill="both", expand=True, pady=6)

        self.train_log = tk.Text(output_frame, height=8, font=("Courier", 10), bg="#0f172a", fg="#38bdf8", padx=8, pady=8)
        self.train_log.pack(fill="both", expand=True)

    def _browse_pdf(self):
        path = filedialog.askopenfilename(
            initialdir=os.path.abspath("data/reference_pdfs"),
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if path:
            self.entry_pdf.delete(0, tk.END)
            self.entry_pdf.insert(0, path)

    def _on_folder_train(self):
        """Train automatically from the reference PDF and curriculum JSON folders."""
        self.lbl_folder_status.config(text="Scanning data/reference_pdfs/ and data/curriculum_json/ ...", fg="#0284c7")
        self.btn_folder_train.config(state="disabled")
        save_path = "data/trained_models/folder_trained_model.pth"

        def run_thread():
            try:
                from trainer import train_offline_model, resolve_paths
                pdf_paths = resolve_paths([], ".pdf", "data/reference_pdfs")
                json_paths = resolve_paths([], ".json", "data/curriculum_json")
                if not pdf_paths and not json_paths:
                    self.lbl_folder_status.config(text="No files found! Add PDFs/JSON to data/ folders first.", fg="#dc2626")
                    return
                generated_json = [
                    _convert_pdf_to_json_for_training(path, "data")
                    for path in pdf_paths
                ]
                stats = train_offline_model(
                    pdf_paths=[],
                    json_paths=json_paths + generated_json,
                    save_path=save_path,
                    epochs=12,
                )
                self.default_model_path = save_path
                self.qa_runner = OfflineQAModelRunner(save_path)
                file_size_mb = os.path.getsize(save_path) / (1024.0 * 1024.0)
                self.lbl_folder_status.config(
                    text=f"✅ Trained from folders! ({file_size_mb:.2f} MB, {stats['num_chunks']} chunks, loss={stats['final_loss']:.4f})",
                    fg="#059669"
                )
                self.notebook.select(self.qa_tab)
                messagebox.showinfo(
                    "Folder Training Complete!",
                    f"Model trained from {len(pdf_paths)} PDF(s) and {len(json_paths)} JSON file(s).\n"
                    f"Saved: {save_path}\n"
                    f"Size: {file_size_mb:.2f} MB | Chunks: {stats['num_chunks']}"
                )
            except Exception as e:
                self.lbl_folder_status.config(text=f"Error: {e}", fg="#dc2626")
                messagebox.showerror("Training Error", str(e))
            finally:
                self.btn_folder_train.config(state="normal")

        threading.Thread(target=run_thread, daemon=True).start()

    def _on_auto_upload_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")])
        if not path:
            return

        self.lbl_auto_status.config(
            text=f"PDF -> TXT -> JSON -> AI training: {os.path.basename(path)}...",
            fg="#0284c7",
        )
        self.btn_auto_upload.config(state="disabled")
        save_path = "models/uploaded_model.pth"

        def run_thread():
            try:
                generated_json = _convert_pdf_to_json_for_training(path, "data")
                stats = train_offline_model(
                    pdf_paths=[],
                    json_paths=[generated_json],
                    save_path=save_path,
                    epochs=12,
                )
                self.default_model_path = save_path
                self.qa_runner = OfflineQAModelRunner(save_path)
                file_size_mb = os.path.getsize(save_path) / (1024.0 * 1024.0)

                self.lbl_auto_status.config(
                    text=f"✅ PDF -> TXT -> JSON -> trained model activated "
                    f"({file_size_mb:.2f} MB, {stats['num_chunks']} chunks)",
                    fg="#059669"
                )
                # Switch to Q&A tab
                self.notebook.select(self.qa_tab)
                messagebox.showinfo(
                    "Model Trained!",
                    f"Machine successfully trained on {os.path.basename(path)}!\nYou can now ask questions about this textbook."
                )
            except Exception as e:
                self.lbl_auto_status.config(text=f"Error: {e}", fg="#dc2626")
                messagebox.showerror("Training Error", str(e))
            finally:
                self.btn_auto_upload.config(state="normal")

        threading.Thread(target=run_thread, daemon=True).start()

    def _browse_json(self):
        path = filedialog.askopenfilename(
            initialdir=os.path.abspath("data/curriculum_json"),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.entry_json.delete(0, tk.END)
            self.entry_json.insert(0, path)

    def _on_train(self):
        pdf_path = self.entry_pdf.get().strip()
        json_path = self.entry_json.get().strip()
        save_path = self.entry_save.get().strip()
        try:
            epochs = int(self.spin_epochs.get())
        except ValueError:
            epochs = 12

        self.btn_train.config(state="disabled", text="⏳ Training...")
        self.train_log.delete("1.0", tk.END)
        self.train_log.insert(tk.END, "Initializing PyTorch training pipeline...\n")

        def run_thread():
            try:
                generated_json = (
                    [_convert_pdf_to_json_for_training(pdf_path, "data")]
                    if pdf_path
                    else []
                )
                stats = train_offline_model(
                    pdf_paths=[],
                    json_paths=([json_path] if json_path else []) + generated_json,
                    save_path=save_path,
                    epochs=epochs,
                )
                file_size_mb = os.path.getsize(save_path) / (1024.0 * 1024.0)
                msg = (
                    f"\nSUCCESSFUL TRAINING RUN!\n"
                    f"Saved Checkpoint: {save_path}\n"
                    f"File Size on Disk: {file_size_mb:.2f} MB\n"
                    f"Parameters: {stats['total_params']:,}\n"
                    f"Curriculum Chunks: {stats['num_chunks']}\n"
                    f"Training Latency: {stats['training_time_s']:.2f} seconds\n"
                    f"Final Loss: {stats['final_loss']:.4f}\n"
                    f"Offline Verified: 100% (No APIs Used)\n"
                )
                self.train_log.insert(tk.END, msg)
                self.default_model_path = save_path
                self.qa_runner = OfflineQAModelRunner(save_path)
            except Exception as e:
                self.train_log.insert(tk.END, f"\nError during training: {e}\n")
            finally:
                self.btn_train.config(state="normal", text="🚀 Start Local Training")

        threading.Thread(target=run_thread, daemon=True).start()

    # ------------------ TAB 3: STUDY AID GENERATOR ------------------
    def _build_aid_tab(self):
        input_frame = tk.LabelFrame(self.aid_tab, text=" Input: 150-300 Word Educational Passage ", font=("Helvetica", 10, "bold"), padx=12, pady=8)
        input_frame.pack(fill="x", pady=6)

        samples_frame = tk.Frame(input_frame)
        samples_frame.pack(fill="x", pady=4)

        tk.Button(samples_frame, text="🌿 Biology Sample", command=lambda: self._load_aid_sample("sample_passages/01_photosynthesis.txt")).pack(side="left", padx=4)
        tk.Button(samples_frame, text="⚡ Physics Sample", command=lambda: self._load_aid_sample("sample_passages/02_renewable_energy.txt")).pack(side="left", padx=4)
        tk.Button(samples_frame, text="💧 Earth Science Sample", command=lambda: self._load_aid_sample("sample_passages/03_water_cycle.txt")).pack(side="left", padx=4)

        self.aid_entry = tk.Text(input_frame, height=5, font=("Helvetica", 10), wrap="word")
        self.aid_entry.pack(fill="x", pady=4)

        tk.Button(
            input_frame,
            text="Generate Summary & 3 Quiz Questions",
            font=("Helvetica", 10, "bold"),
            bg="#6366f1",
            fg="white",
            padx=12,
            pady=4,
            command=self._on_generate_aid,
        ).pack(anchor="w", pady=4)

        # Output Section
        output_frame = tk.LabelFrame(self.aid_tab, text=" Output: Study Aids & Telemetry ", font=("Helvetica", 10, "bold"), padx=12, pady=8)
        output_frame.pack(fill="both", expand=True, pady=6)

        self.aid_output = tk.Text(output_frame, font=("Courier", 10), wrap="word", bg="#f1f5f9", padx=8, pady=8)
        self.aid_output.pack(fill="both", expand=True)

    def _load_aid_sample(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.aid_entry.delete("1.0", tk.END)
                self.aid_entry.insert(tk.END, f.read().strip())

    def _on_generate_aid(self):
        passage = self.aid_entry.get("1.0", tk.END).strip()
        if not passage:
            return

        from engine import OfflineDeterministicEngine
        engine = OfflineDeterministicEngine()
        out_text, telemetry = engine.generate(passage)

        self.aid_output.delete("1.0", tk.END)
        self.aid_output.insert(tk.END, out_text + "\n\n" + telemetry.format_report())


def run_desktop_gui():
    root = tk.Tk()
    app = OfflineLearningAssistantDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_desktop_gui()
