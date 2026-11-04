#!/usr/bin/env python3
"""
Offline Neural Learning Assistant (Training & Inference CLI)
Enables:
1. Training a PyTorch neural model directly on PDF textbooks and JSON datasets.
2. Compressing the model and knowledge representations into a .pth file.
3. Answering questions offline based strictly on the trained knowledge (0 external APIs).
"""

import argparse
import os
import sys

from generate_sample_data import create_sample_curriculum_json, create_sample_textbook_pdf
from qa_engine import OfflineQAModelRunner
from trainer import train_offline_model

# Terminal styling
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


def print_header():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════════════╗
║        OFFLINE NEURAL LEARNING ASSISTANT (TRAIN & ANSWER ENGINE)         ║
║          Train on PDFs & JSON • Zero APIs • Compressed .pth Model        ║
╚══════════════════════════════════════════════════════════════════════════╝{RESET}
""")


def cmd_train(args):
    print(f"{BOLD}[Mode] Training Edge Neural Model on local documents...{RESET}")
    stats = train_offline_model(
        pdf_paths=args.pdf,
        json_paths=args.json,
        save_path=args.save,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"{GREEN}{BOLD}Training Successful! Model saved to:{RESET} {args.save}")
    print(f"You can now query this model anytime offline using:")
    print(f"  python train_and_qa.py ask --model {args.save} --question \"Your question here\"")


def cmd_ask(args):
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found. Please train first or run demo.", file=sys.stderr)
        sys.exit(1)

    runner = OfflineQAModelRunner(args.model)
    response = runner.answer(args.question)
    print(response.format_card())


def cmd_interactive(args):
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found. Please train first or run demo.", file=sys.stderr)
        sys.exit(1)

    print(f"{BOLD}[Interactive Offline Q&A Mode]{RESET}")
    print(f"Loaded Model: {YELLOW}{args.model}{RESET}")
    runner = OfflineQAModelRunner(args.model)
    print(f"{GREEN}Ready! Ask any question based on the trained PDF/JSON curriculum.{RESET}")
    print(f"Type 'exit' or press Ctrl+C to quit.\n")

    while True:
        try:
            q = input(f"{BOLD}{CYAN}Student Question > {RESET}").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit", "q"}:
                print("Exiting.")
                break
            response = runner.answer(q)
            print(response.format_card())
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def cmd_demo(args):
    print(f"{BOLD}[Demo Mode] Starting full automated end-to-end pipeline:{RESET}")
    print("1. Generating sample educational science textbook (PDF)")
    print("2. Generating structured curriculum Q&A dataset (JSON)")
    print("3. Training PyTorch neural model locally from scratch")
    print("4. Compressing and saving to .pth model file")
    print("5. Answering sample questions offline without any API\n")

    pdf_file = "sample_data/science_textbook.pdf"
    json_file = "sample_data/curriculum_qa.json"
    model_file = "models/demo_curriculum_model.pth"

    create_sample_textbook_pdf(pdf_file)
    create_sample_curriculum_json(json_file)

    # Train model
    train_offline_model(
        pdf_paths=[pdf_file],
        json_paths=[json_file],
        save_path=model_file,
        epochs=12,
        batch_size=4,
    )

    # Run sample questions
    print(f"\n{BOLD}{GREEN}=== TESTING TRAINED .PTH MODEL OFFLINE (ZERO APIs) ==={RESET}")
    runner = OfflineQAModelRunner(model_file)

    test_questions = [
        "What is chlorophyll and where is it found in plant cells?",
        "What is the function of stomata in plant leaves?",
        "How do solar photovoltaic cells generate electric current?",
    ]

    for q in test_questions:
        resp = runner.answer(q)
        print(resp.format_card())


def main():
    parser = argparse.ArgumentParser(
        description="Offline Neural Learning Assistant: Train on PDF/JSON and Answer Questions"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    # Train subcommand
    train_p = subparsers.add_parser("train", help="Train neural model on PDF and JSON files")
    train_p.add_argument("--pdf", nargs="+", help="Path to one or more PDF textbook files")
    train_p.add_argument("--json", nargs="+", help="Path to one or more JSON Q&A / curriculum files")
    train_p.add_argument("--save", type=str, default="models/offline_model.pth", help="Target .pth model path")
    train_p.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    train_p.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    train_p.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    # Ask subcommand
    ask_p = subparsers.add_parser("ask", help="Ask a question to the trained .pth model")
    ask_p.add_argument("--model", type=str, default="models/offline_model.pth", help="Path to trained .pth model")
    ask_p.add_argument("--question", "-q", type=str, required=True, help="Question text")

    # Interactive subcommand
    inter_p = subparsers.add_parser("interactive", help="Start interactive terminal Q&A session")
    inter_p.add_argument("--model", type=str, default="models/offline_model.pth", help="Path to trained .pth model")

    # Demo subcommand
    demo_p = subparsers.add_parser("demo", help="Run automated end-to-end training and Q&A demonstration")

    args = parser.parse_args()
    print_header()

    if args.subcommand == "train":
        if not args.pdf and not args.json:
            print("Error: Specify at least one --pdf or --json file to train.", file=sys.stderr)
            sys.exit(1)
        cmd_train(args)
    elif args.subcommand == "ask":
        cmd_ask(args)
    elif args.subcommand == "interactive":
        cmd_interactive(args)
    elif args.subcommand == "demo" or args.subcommand is None:
        cmd_demo(args)


if __name__ == "__main__":
    main()

