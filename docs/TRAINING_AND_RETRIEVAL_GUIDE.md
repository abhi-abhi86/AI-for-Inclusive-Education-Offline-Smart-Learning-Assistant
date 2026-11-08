# Training and Fast Retrieval Guide

This project is an offline retrieval-and-explanation system. It does not train a
large language model from scratch. Training teaches a small local Transformer
how to match questions to textbook passages; the final explanation is composed
from the retrieved textbook evidence.

## 1. End-to-end data flow

```text
PDF
  -> page-preserving TXT
  -> structured JSON
  -> training chunks + generated question/answer pairs
  -> neural embeddings
  -> .pth checkpoint
  -> question retrieval
  -> grounded explanation
```

Uploaded books are isolated under:

```text
data/uploaded_documents/<document-id>/
  pdf/<book>.pdf
  text/<book>.txt
  json/<book>_knowledge.json
models/uploaded_documents/<document-id>.pth
```

The TXT file is the canonical intermediate source. JSON is generated from TXT,
not directly from PDF.

## 2. How text becomes retrieval chunks

Each extracted page is retained as a page record. Long PDF pages are split into
overlapping word windows by `extract_text_from_pdf()`:

- Default window: 120 words
- Default overlap: 25 words
- Step: `chunk_size - overlap`
- Every chunk keeps `source_file` and `page_or_id`

The overlap prevents an answer from being broken at a window boundary. Keep
chunks large enough to contain a complete definition or explanation, but small
enough that unrelated topics do not share one embedding.

Recommended chunk rules:

| Content | Recommended chunk |
|---|---:|
| Definition or short fact | 60–120 words |
| Process or mechanism | 100–180 words |
| Worked example | One complete example |
| Table/list | Keep the heading and all rows together |
| Figure caption | Keep with the paragraph that explains it |

Never remove the page number, heading, section, or source filename. These fields
are required for citations and debugging incorrect answers.

## 3. JSON record design

The generated archive contains:

```json
{
  "page_or_id": "Page 3, Section 2",
  "text": "A complete evidence passage...",
  "source_file": "biology.pdf",
  "document_metadata": {
    "title": "Heredity"
  }
}
```

Generated Q&A records add:

```json
{
  "question": "What is heredity?",
  "answer": "Heredity is...",
  "question_type": "definition",
  "source_file": "biology.pdf",
  "page_or_id": "Page 3"
}
```

The trainer loads both passage records and Q&A records. This gives retrieval
enough evidence to answer exact questions while still supporting questions that
were not generated in advance.

## 4. Training a model

### Train from the standard PDF/JSON folders

```bash
python train_and_qa.py train \
  --pdf pdfs_to_train \
  --json data/curriculum_json \
  --save models/curriculum_model.pth \
  --epochs 15
```

### Train from one converted uploaded book

Use the GUI:

1. Select the PDF.
2. Wait for PDF -> TXT -> JSON to finish.
3. Confirm **Train AI from this JSON**.
4. Wait for the training completion message.
5. Click **Ask Questions on this Textbook**.

The GUI sends only the generated JSON path to training. The resulting checkpoint
contains:

- model weights
- vocabulary
- all training chunks
- generated Q&A pairs
- one normalized embedding per chunk
- training statistics

## 5. How fast retrieval works

At training time, each chunk is encoded once and saved in
`knowledge_embeddings`. At question time:

1. The question is normalized and encoded once.
2. Cosine similarity ranks neural semantic matches.
3. Content words are matched lexically after removing question stop words.
4. Singular/plural forms are normalized.
5. Exact generated-question matches receive priority.
6. Reciprocal Rank Fusion combines neural and lexical rankings.
7. The best passage is selected and the next matching passages from the same
   source are used as supporting evidence.

This avoids rescanning and re-encoding the full textbook for every question.
The response includes the source filename, page reference, confidence, and
context passage.

## 6. How explanations are produced

The answer synthesizer is grounded in the selected passage:

1. Use an explicit generated JSON answer when one exists.
2. Otherwise choose the highest-overlap evidence sentence.
3. Add nearby supporting sentences from the same source.
4. Resolve simple pronouns when the previous sentence provides the subject.
5. Add structured highlights for definitions, conversions, absorption, release,
   and power-generation statements.
6. Refuse or report “not found” when retrieval confidence is insufficient.

For best explanations, keep each chunk focused on one topic and include the
section heading in the chunk text.

## 7. Improving answers for the same question

When a student asks the same question in different forms, add or preserve:

- an exact generated question
- a short direct answer
- the full supporting passage
- alternate terms and singular/plural forms in the passage
- the page and section reference

Example:

```json
{
  "question": "What is mass?",
  "answer": "Mass is the amount of matter in an object.",
  "text": "Mass is the amount of matter in an object. It is measured in kilograms.",
  "page_or_id": "Page 12, Section 1"
}
```

This supports:

- “What is mass?”
- “Define mass.”
- “What do we mean by mass?”
- “How is mass measured?”

Do not add answers that are not supported by the TXT source. The quality
detector should reject weak or unrelated generated pairs before training.

## 8. Validation checklist

Before deploying a model:

```bash
python -m py_compile data_loader.py trainer.py qa_engine.py
python train_and_qa.py check --model models/curriculum_model.pth
```

Test at least:

- one exact generated question
- one paraphrased question
- one question requiring a page citation
- one question requiring two nearby sentences
- one unrelated question that should return “not found”

If answers are vague, inspect the JSON first. Incorrect TXT extraction,
incorrect page boundaries, or weak generated Q&A records must be fixed before
increasing epochs.
