#!/usr/bin/env python3
"""Convert the PDF training folder into structured JSON archives."""

import argparse

from data_loader import convert_pdf_folder_to_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDFs to page-aware JSON knowledge archives."
    )
    parser.add_argument("--pdf-root", default="pdfs_to_train")
    parser.add_argument("--json-root", default="data/json_from_pdfs")
    args = parser.parse_args()

    result = convert_pdf_folder_to_json(args.pdf_root, args.json_root)
    print(
        f"Converted {result['converted_count']}/{result['pdf_count']} PDFs "
        f"to {result['json_root']}"
    )
    for failure in result["failures"]:
        print(f"Skipped {failure['pdf_path']}: {failure['error']}")


if __name__ == "__main__":
    main()
