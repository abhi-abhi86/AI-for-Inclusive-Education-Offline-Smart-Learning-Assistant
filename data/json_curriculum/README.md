# 📄 JSON Curriculum & Q&A Folder (`data/json_curriculum/`)

Place all your structured curriculum Q&A pairs, flashcards, and topic notes in **JSON format (`.json`)** in this directory.

---

### Supported JSON Formats:

#### 1. Question-Answer Pairs (Recommended for Exam Prep):
```json
[
  {
    "question": "What is the function of stomata?",
    "answer": "Stomata are microscopic leaf pores that regulate carbon dioxide intake and oxygen release.",
    "context": "Photosynthesis gas exchange in plants."
  },
  {
    "question": "What is chlorophyll?",
    "answer": "Chlorophyll is the green pigment in chloroplasts that absorbs sunlight.",
    "context": "Plant cell biology."
  }
]
```

#### 2. Key-Value Topic Dictionary:
```json
{
  "Photosynthesis": "Green plants convert solar light energy into chemical energy stored as glucose.",
  "Renewable Energy": "Energy from naturally replenishing resources like solar, wind, and hydroelectric."
}
```

---

### How to Train:
1. Drop any `.json` file here.
2. Run:
   ```bash
   ./venv/bin/python3 train_and_qa.py train
   ```
   All JSON files in this directory will be automatically ingested and trained into your `.pth` model!

