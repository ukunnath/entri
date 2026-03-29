"""
main.py
=======
Console-based entry point for the AI-Powered Paraphrasing Tool.

Usage
-----
Interactive mode (prompts for input):
    python main.py

Batch mode (read sentences from a text file, one per line):
    python main.py --file sample_texts.txt

Single sentence mode:
    python main.py --text "The quick brown fox jumps over the lazy dog."

All modes support optional flags:
    --no-eval     skip evaluation metrics (faster)
    --no-quality  skip quality checks
    --candidates N  number of paraphrase candidates (default 3)
    --verbose     show detailed logs
"""

import argparse
import sys
import textwrap
from typing import List

from paraphraser    import Paraphraser
from quality_checker import QualityChecker
from evaluator       import Evaluator

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    text: str,
    paraphraser: Paraphraser,
    checker:     QualityChecker,
    evaluator:   Evaluator,
    n_candidates: int = 3,
    run_eval:    bool = True,
    run_quality: bool = True,
) -> None:

    sep = "─" * 62

    print(f"\n{sep}")
    print("  INPUT TEXT")
    print(sep)
    for line in textwrap.wrap(text, width=60):
        print(f"  {line}")

    # 1. Generate paraphrases
    print(f"\n  Generating {n_candidates} paraphrase candidate(s) …")
    candidates = paraphraser.paraphrase(text, num_return_sequences=n_candidates)

    print(f"\n{sep}")
    print("PARAPHRASE CANDIDATES")
    print(sep)
    for i, p in enumerate(candidates, 1):
        print(f"\n  [{i}] ", end="")
        print(textwrap.fill(p, width=58, subsequent_indent="      "))

    # 2. Quality checks
    if run_quality:
        print(f"\n{sep}")
        print("  QUALITY CHECKS")
        print(sep)
        quality_reports = []
        for i, p in enumerate(candidates, 1):
            print(f"\n  Candidate #{i}")
            qr = checker.check(p)
            print(qr.summary())
            quality_reports.append(qr)

    #3. Evaluation metrics
    if run_eval:
        print(f"\n{sep}")
        print("  EVALUATION METRICS")
        eval_report = evaluator.evaluate(text, candidates)
        print(eval_report.summary())

        # Highlight the best candidate
        best = evaluator.pick_best(eval_report)
        if best:
            print(f"\n  ★  BEST CANDIDATE (highest quality score):")
            print(f"     {best.paraphrase}\n")

    print(f"{sep}\n")



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI-Powered Paraphrasing Tool (console)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--text",       type=str,  default=None,
                   help="Single input text to paraphrase.")
    p.add_argument("--file",       type=str,  default=None,
                   help="Path to a text file (one sentence/paragraph per line).")
    p.add_argument("--candidates", type=int,  default=3,
                   help="Number of paraphrase candidates to generate (default: 3).")
    p.add_argument("--no-eval",    action="store_true",
                   help="Skip evaluation metrics.")
    p.add_argument("--no-quality", action="store_true",
                   help="Skip quality checks.")
    p.add_argument("--verbose",    action="store_true",
                   help="Show detailed loading logs.")
    return p



def main() -> None:

    args = build_parser().parse_args()
    paraphraser = Paraphraser(verbose=args.verbose)
    checker     = QualityChecker(verbose=args.verbose) if not args.no_quality else None
    evaluator   = Evaluator(verbose=args.verbose)      if not args.no_eval    else None

    run_quality = not args.no_quality and checker  is not None
    run_eval    = not args.no_eval    and evaluator is not None

    def process(text: str) -> None:
        run_pipeline(
            text         = text,
            paraphraser  = paraphraser,
            checker      = checker     or QualityChecker(verbose=False),
            evaluator    = evaluator   or Evaluator(verbose=False),
            n_candidates = args.candidates,
            run_eval     = run_eval,
            run_quality  = run_quality,
        )

    #Dispatch
    if args.text:
        process(args.text)

    elif args.file:
        try:
            with open(args.file, encoding="utf-8") as fh:
                lines = [l.strip() for l in fh if l.strip()]
            print(f"  Loaded {len(lines)} input(s) from '{args.file}'.")
            for line in lines:
                process(line)
        except FileNotFoundError:
            print(f"  ERROR: File not found — '{args.file}'")
            sys.exit(1)

    else:
        # Interactive REPL
        print("  Interactive mode — type a sentence and press Enter.")
        print("  Type 'quit' or press Ctrl+C to exit.\n")
        while True:
            try:
                text = input("  ➤  Enter text: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  Goodbye!")
                break
            if not text:
                continue
            if text.lower() in {"quit", "exit", "q"}:
                print("  Goodbye!")
                break
            process(text)


if __name__ == "__main__":
    main()
