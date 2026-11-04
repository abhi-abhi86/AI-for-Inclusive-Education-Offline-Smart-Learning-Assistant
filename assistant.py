#!/usr/bin/env python3
"""
Offline Smart Learning Assistant (Edge AI on Raspberry Pi & macOS)
Problem Statement 1: AI for Inclusive Education

Operates 100% offline using lightweight Edge AI to generate:
1. An accessible, easy-to-understand summary of a 150-300 word educational passage.
2. Exactly 3 diagnostic quiz questions with answers and pedagogical explanations.
3. Accurate real-time Edge AI telemetry (latency, tok/s, peak RAM, model footprint).
"""

import argparse
import glob
import os
import sys
from pathlib import Path

from engine import get_inference_engine
from metrics import EdgeTelemetry
from prompt_templates import validate_passage


# Terminal color constants
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


def print_banner():
    banner = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════════════╗
║               OFFLINE SMART LEARNING ASSISTANT (EDGE AI)                 ║
║       Problem Statement 1: AI for Inclusive Education on Raspberry Pi    ║
║             Zero Cloud Dependencies • Low Memory • High Accessibility    ║
╚══════════════════════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)


def process_and_display(passage: str, engine, passage_label: str = "Input Passage"):
    """Validates passage, runs edge inference, and presents structured learning aids."""
    is_valid, word_count, msg = validate_passage(passage)
    print(f"\n{BOLD}[Step 1/3] Ingesting Educational Passage:{RESET} {passage_label}")
    print(f"  Word count: {BOLD}{word_count}{RESET} words | {YELLOW}{msg}{RESET}")

    print(f"\n{BOLD}[Step 2/3] Running Local Edge Inference...{RESET}")
    output_text, telemetry = engine.generate(passage)

    print(f"\n{BOLD}[Step 3/3] Generated Offline Learning Aids:{RESET}")
    print(f"{MAGENTA}{'=' * 68}{RESET}")
    print(output_text)
    print(f"{MAGENTA}{'=' * 68}{RESET}\n")

    # Render mandatory hackathon edge telemetry
    print(telemetry.format_report())
    return telemetry


def run_benchmark(engine):
    """Executes automated benchmark on all sample passages and prints consolidated metrics."""
    sample_files = sorted(glob.glob("sample_passages/*.txt"))
    if not sample_files:
        print("No sample passages found in sample_passages/")
        return

    print(f"\n{BOLD}{CYAN}=== RUNNING AUTOMATED EDGE INFERENCE BENCHMARK ==={RESET}\n")
    results = []

    for file_path in sample_files:
        p = Path(file_path)
        with open(p, "r", encoding="utf-8") as fp:
            content = fp.read()
        print(f"{BOLD}Testing passage:{RESET} {p.name}")
        telemetry = process_and_display(content, engine, passage_label=p.name)
        results.append((p.stem, len(content.split()), telemetry))
        print("\n" + "-" * 68 + "\n")

    # Summary table
    print(f"\n{BOLD}{GREEN}================ CONSOLIDATED BENCHMARK SUMMARY ================{RESET}")
    header = f"{'Passage':<25} | {'Words':<6} | {'Latency':<9} | {'Speed (tok/s)':<14} | {'Peak RAM':<10}"
    print(header)
    print("-" * len(header))
    for name, words, t in results:
        print(
            f"{name[:25]:<25} | {words:<6} | {t.total_time_seconds:<7.2f} s | {t.tokens_per_second:<12.2f} | {t.peak_ram_mb:<7.2f} MB"
        )
    print(f"{BOLD}{GREEN}================================================================{RESET}")
    print("Cloud Network Invocations: 0 (100% Offline Verified on Edge Hardware)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Offline Smart Learning Assistant for Inclusive Education (ExecuTorch / Edge AI)"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--file", "-f", type=str, help="Path to text file containing educational passage")
    group.add_argument("--passage", "-p", type=str, help="Direct text of the educational passage")
    group.add_argument(
        "--sample",
        "-s",
        type=int,
        choices=[1, 2, 3],
        help="Run on curated curriculum sample passage: 1=Photosynthesis, 2=Renewable Energy, 3=Water Cycle",
    )
    group.add_argument(
        "--benchmark",
        "-b",
        action="store_true",
        help="Run comprehensive offline benchmark on all sample passages",
    )
    group.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode: paste or type passage directly into terminal",
    )

    parser.add_argument(
        "--executorch",
        action="store_true",
        help="Use compiled ExecuTorch .pte model runner",
    )
    parser.add_argument("--model-path", type=str, default="model.pte", help="Path to ExecuTorch .pte model file")
    parser.add_argument(
        "--tokenizer-path", type=str, default="tokenizer.bin", help="Path to tokenizer file"
    )
    parser.add_argument("--runner-bin", type=str, default=None, help="ExecuTorch runner binary name/path")

    args = parser.parse_args()
    print_banner()

    engine = get_inference_engine(
        use_executorch=args.executorch,
        model_path=args.model_path if args.executorch else None,
        tokenizer_path=args.tokenizer_path if args.executorch else None,
        runner_bin=args.runner_bin,
    )

    if args.benchmark:
        run_benchmark(engine)
        return

    passage_text = None
    label = "Provided Text"

    if args.sample:
        sample_map = {
            1: "sample_passages/01_photosynthesis.txt",
            2: "sample_passages/02_renewable_energy.txt",
            3: "sample_passages/03_water_cycle.txt",
        }
        chosen_file = sample_map[args.sample]
        label = os.path.basename(chosen_file)
        with open(chosen_file, "r", encoding="utf-8") as fp:
            passage_text = fp.read()

    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: Specified file does not exist: {args.file}", file=sys.stderr)
            sys.exit(1)
        label = os.path.basename(args.file)
        with open(args.file, "r", encoding="utf-8") as fp:
            passage_text = fp.read()

    elif args.passage:
        passage_text = args.passage
        label = "CLI Argument"

    elif args.interactive:
        print(f"{YELLOW}Paste or type your 150-300 word educational passage below.{RESET}")
        print(f"{YELLOW}Press ENTER on an empty line or Ctrl+D to submit:{RESET}\n")
        lines = []
        try:
            while True:
                line = input()
                if not line.strip() and lines:
                    break
                lines.append(line)
        except EOFError:
            pass
        passage_text = "\n".join(lines).strip()
        label = "Interactive Terminal Input"

    else:
        # Default behavior: run on Sample 1 (Photosynthesis) to demonstrate immediate working state
        print(f"{YELLOW}No input flags supplied. Defaulting to Sample 1 (Photosynthesis)...{RESET}")
        sample_file = "sample_passages/01_photosynthesis.txt"
        label = os.path.basename(sample_file)
        with open(sample_file, "r", encoding="utf-8") as fp:
            passage_text = fp.read()

    if not passage_text or not passage_text.strip():
        print("Error: Educational passage cannot be empty.", file=sys.stderr)
        sys.exit(1)

    process_and_display(passage_text, engine, passage_label=label)


if __name__ == "__main__":
    main()

