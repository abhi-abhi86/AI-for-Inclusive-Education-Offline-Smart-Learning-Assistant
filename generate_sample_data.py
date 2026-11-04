"""
Utility to create a sample educational curriculum PDF for testing the offline trainer.
Constructs a valid PDF from scratch in pure Python (no third-party dependencies required).
"""

import os


def create_sample_textbook_pdf(output_path: str = "sample_data/science_textbook.pdf"):
    """Creates a 2-page educational science textbook PDF."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    page1_text = (
        "CHAPTER 1: CELL BIOLOGY AND PHOTOSYNTHESIS\\n\\n"
        "Photosynthesis is the fundamental biological process by which green plants and algae "
        "convert light energy into chemical energy stored as glucose. Plants absorb sunlight using "
        "chlorophyll located inside chloroplasts. They absorb carbon dioxide from the air through "
        "microscopic leaf pores called stomata, while roots absorb water from soil. "
        "Powered by solar energy, chloroplasts convert carbon dioxide and water into glucose. "
        "Oxygen is released back into the atmosphere as a vital byproduct through stomata."
    )

    page2_text = (
        "CHAPTER 2: RENEWABLE ENERGY AND PHYSICS\\n\\n"
        "Renewable energy originates from naturally replenishing resources such as solar, wind, and hydro. "
        "Solar photovoltaic cells convert incident photons directly into electric current. "
        "Wind turbines capture the kinetic energy of air currents to spin electric generators. "
        "Hydroelectric power harnesses gravitational potential energy of falling water. "
        "A key challenge is intermittency, which requires battery storage systems to balance electrical grids."
    )

    # Minimal standard PDF stream generator
    # We construct 2 pages with standard PDF syntax
    def escape_pdf(text):
        return text.replace("(", "\\(").replace(")", "\\)")

    lines_p1 = page1_text.split("\\n")
    p1_stream_lines = ["BT /F1 12 Tf 50 750 Td 16 TL"]
    for l in lines_p1:
        if l:
            p1_stream_lines.append(f"({escape_pdf(l)}) Tj T*")
        else:
            p1_stream_lines.append("T*")
    p1_stream_lines.append("ET")
    p1_content = "\n".join(p1_stream_lines)

    lines_p2 = page2_text.split("\\n")
    p2_stream_lines = ["BT /F1 12 Tf 50 750 Td 16 TL"]
    for l in lines_p2:
        if l:
            p2_stream_lines.append(f"({escape_pdf(l)}) Tj T*")
        else:
            p2_stream_lines.append("T*")
    p2_stream_lines.append("ET")
    p2_content = "\n".join(p2_stream_lines)

    pdf = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Resources << /Font << /F1 7 0 R >> >> >>
endobj
4 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >>
endobj
5 0 obj
<< /Length {len(p1_content)} >>
stream
{p1_content}
endstream
endobj
6 0 obj
<< /Length {len(p2_content)} >>
stream
{p2_content}
endstream
endobj
7 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 8
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000124 00000 n 
0000000250 00000 n 
0000000376 00000 n 
0000000476 00000 n 
0000000576 00000 n 
trailer
<< /Size 8 /Root 1 0 R >>
startxref
650
%%EOF
"""
    with open(output_path, "wb") as f:
        f.write(pdf.encode("latin-1"))

    print(f"Created sample educational PDF: {output_path}")
    return output_path


def create_sample_curriculum_json(output_path: str = "sample_data/curriculum_qa.json"):
    """Creates a sample curriculum Q&A JSON dataset."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    import json

    data = [
        {
            "question": "What is chlorophyll and where is it found in plant cells?",
            "answer": "Chlorophyll is the green pigment located inside chloroplasts that absorbs sunlight for photosynthesis.",
            "context": "Plant cells contain chloroplasts which house the green photosynthetic pigment chlorophyll."
        },
        {
            "question": "What is the function of stomata in leaves?",
            "answer": "Stomata are microscopic pores in leaves that regulate the intake of carbon dioxide and release of oxygen.",
            "context": "Stomata on the underside of plant leaves facilitate gas exchange necessary for photosynthesis."
        },
        {
            "question": "What are the primary products of photosynthesis?",
            "answer": "The primary products are glucose (which fuels plant growth) and oxygen gas (released as a byproduct).",
            "context": "During photosynthesis, carbon dioxide and water react in sunlight to yield glucose and oxygen."
        },
        {
            "question": "How do photovoltaic cells generate electricity?",
            "answer": "Photovoltaic cells convert incident photons from sunlight directly into electrical current via the photoelectric effect.",
            "context": "Solar photovoltaic technology utilizes semiconductor panels to convert sunlight directly into DC electricity."
        },
        {
            "question": "What is the hydrological cycle driven by?",
            "answer": "The hydrological water cycle is driven continuously by solar thermal energy and Earth's gravity.",
            "context": "Solar heat causes evaporation while gravity causes precipitation, runoff, and groundwater infiltration."
        }
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Created sample curriculum JSON: {output_path}")
    return output_path


if __name__ == "__main__":
    create_sample_textbook_pdf()
    create_sample_curriculum_json()

