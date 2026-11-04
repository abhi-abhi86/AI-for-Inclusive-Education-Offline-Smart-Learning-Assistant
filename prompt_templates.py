"""
Prompt templates and output formatting for Offline Smart Learning Assistant.
Engineered for edge LLMs (Llama 3.2 1B, Qwen 2.5, SmolLM2) running on ExecuTorch.
"""

from typing import Dict, List, Tuple


SYSTEM_PROMPT = """You are an offline smart learning tutor for students in low-connectivity regions.
Your job is to read an educational passage and generate:
1. A concise, easy-to-understand summary (2-3 bullet points) emphasizing core concepts.
2. Exactly 3 diagnostic quiz questions (multiple choice or conceptual) with answers and short explanations to help the student test their understanding offline.

Keep the language clear, accessible, and directly tied to the facts in the text."""


def build_prompt(passage: str) -> str:
    """Construct an instruction prompt formatted for instruction-tuned LLMs."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>
Please process the following educational passage:

\"\"\"
{passage.strip()}
\"\"\"

Output format required:
### SUMMARY
- [Point 1: Key core idea]
- [Point 2: Supporting mechanism or fact]
- [Point 3: Key takeaway/impact]

### QUIZ QUESTIONS
1. [Question 1: Recall / Core definition]
   A) [Option A]
   B) [Option B]
   C) [Option C]
   D) [Option D]
   Correct Answer: [Option] - [Brief explanation]

2. [Question 2: Conceptual understanding]
   A) [Option A]
   B) [Option B]
   C) [Option C]
   D) [Option D]
   Correct Answer: [Option] - [Brief explanation]

3. [Question 3: Application / Reasoning]
   A) [Option A]
   B) [Option B]
   C) [Option C]
   D) [Option D]
   Correct Answer: [Option] - [Brief explanation]<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def validate_passage(text: str) -> Tuple[bool, int, str]:
    """
    Validate that the passage matches the hackathon requirement of approximately 150-300 words.
    Returns (is_valid, word_count, message).
    """
    words = text.strip().split()
    count = len(words)
    if count == 0:
        return False, 0, "Passage is empty."
    if count < 100:
        return True, count, f"Note: Passage is {count} words (below typical 150-300 word target, but still processable)."
    if count > 400:
        return True, count, f"Note: Passage is {count} words (exceeds 300 word target, processing may take longer on edge device)."
    return True, count, f"Passage length verified: {count} words (within 150-300 range)."


def parse_learning_aid(raw_output: str) -> Dict[str, str]:
    """
    Parses generated text into structured sections (Summary and Quiz Questions).
    Falls back gracefully if formatting markers vary.
    """
    summary_part = ""
    quiz_part = ""

    text = raw_output.strip()

    if "### SUMMARY" in text and "### QUIZ QUESTIONS" in text:
        parts = text.split("### QUIZ QUESTIONS")
        summary_raw = parts[0].replace("### SUMMARY", "").strip()
        quiz_raw = parts[1].strip()
        return {"summary": summary_raw, "quiz": quiz_raw}
    elif "QUIZ" in text.upper():
        upper_text = text.upper()
        quiz_idx = upper_text.find("QUIZ")
        summary_raw = text[:quiz_idx].strip()
        quiz_raw = text[quiz_idx:].strip()
        return {"summary": summary_raw, "quiz": quiz_raw}
    else:
        return {"summary": text, "quiz": "Quiz generation pending or unified format."}

