#!/usr/bin/env python3
"""Extract every training PDF into a page-preserving UTF-8 text file."""

import argparse

from data_loader import convert_pdf_folder_to_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract PDFs to page-preserving text files."
    )
    parser.add_argument("--pdf-root", default="pdfs_to_train")
    parser.add_argument("--text-root", default="data/converted_text")
    args = parser.parse_args()

    result = convert_pdf_folder_to_text(args.pdf_root, args.text_root)
    print(
        f"Extracted {result['converted_count']}/{result['pdf_count']} PDFs "
        f"to {result['text_root']}"
    )
    for failure in result["failures"]:
        print(f"Skipped {failure['pdf_path']}: {failure['error']}")


if __name__ == "__main__":
    main()
