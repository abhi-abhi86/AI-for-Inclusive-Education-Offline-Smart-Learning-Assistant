#!/usr/bin/env python3
"""Convert the TXT archives created from PDFs into structured JSON archives."""

import argparse

from data_loader import convert_pdf_folder_to_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert page-preserving TXT archives to JSON."
    )
    parser.add_argument("--pdf-root", default="pdfs_to_train")
    parser.add_argument("--json-root", default="data/converted_json")
    parser.add_argument("--text-root", default="data/converted_text")
    args = parser.parse_args()

    result = convert_pdf_folder_to_json(
        args.pdf_root,
        args.json_root,
        text_root=args.text_root,
    )
    print(
        f"Converted {result['converted_count']}/{result['pdf_count']} PDFs "
        f"to {result['json_root']}"
    )
    for failure in result["failures"]:
        print(f"Skipped {failure['pdf_path']}: {failure['error']}")


if __name__ == "__main__":
    main()
