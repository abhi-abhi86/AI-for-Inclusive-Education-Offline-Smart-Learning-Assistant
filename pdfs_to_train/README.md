# PDFs to train

Put text-based PDF textbooks, notes, or study documents in this folder.

The pipeline is always:

**PDF -> page-preserving TXT -> JSON -> offline AI training -> `.pth` answer model**

Train the offline model from every PDF in the folder with:

```bash
./venv/bin/python train_and_qa.py train-from-folders
```

Or start the GUI and choose **Train from PDFs in folder**.

The PDF is converted to a page-preserving TXT archive before JSON conversion.
Scanned/image-only PDFs need an OCR text layer first, for example:

```bash
ocrmypdf input.pdf output.pdf
```
