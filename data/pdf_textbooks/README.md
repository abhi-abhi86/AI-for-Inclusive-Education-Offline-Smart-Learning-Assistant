# 📚 PDF Textbooks Folder (`data/pdf_textbooks/`)

Place all your student semester textbooks, lecture slides, chapter notes, and syllabus documents in **PDF format (`.pdf`)** in this directory.

---

### How It Works:
1. Drop any `.pdf` files here (e.g., `biology_grade10.pdf`, `physics_semester1.pdf`, `chemistry_notes.pdf`).
2. Run the one-click training command:
   ```bash
   ./venv/bin/python3 train_and_qa.py train
   ```
   *(Or click **"⚡ One-Click Train from Folders"** in the Web GUI / Desktop App).*
3. The trainer will automatically scan this folder, extract all text page-by-page, train the PyTorch neural model, and save the compressed `.pth` checkpoint.

