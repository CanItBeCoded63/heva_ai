"""
Convenient one-click CLI entry point to run the full Adversarial Evaluation Harness.

Executes:
1. Ground truth dataset verification & FAISS index building
2. Offline adversarial suite mutation (paraphrased, injected_context, subtle_error, edge_case)
3. End-to-end evaluation runner with claim-by-claim hallucination detection
4. Confidence calibration calculation and reliability diagram plot generation (calibration_curve.png)
5. Behavioral failure mode clustering (KMeans k=3)
6. Automated Markdown evaluation report generation (EVALUATION_REPORT.md)
"""

import os
import sys

# ── Env isolation ─────────────────────────────────────────────────────────────
# The azure-foundry PowerShell profile injects AZURE_AI_* vars into every shell
# session. Clear them BEFORE loading .env so our OpenRouter key always wins.
for _stale_key in ("AZURE_AI_API_KEY", "AZURE_AI_ENDPOINT", "AZURE_AI_MODEL",
                   "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
                   "AZURE_OPENAI_API_VERSION"):
    os.environ.pop(_stale_key, None)
# ─────────────────────────────────────────────────────────────────────────────

# Load .env FIRST with override=True so project keys take priority over any shell env (e.g. Azure Foundry)
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)


import argparse
from generate_dataset import scaffold_dataset
from evaluation.adversarial import build_and_save_adversarial_suites
from evaluation.runner import EvaluationHarness
from report import generate_markdown_report


def run_full_pipeline(seed: int = 42, no_rewriter: bool = False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_p = os.path.join(base_dir, "data", "corpus.json")
    dataset_p = os.path.join(base_dir, "data", "dataset.jsonl")
    suites_p = os.path.join(base_dir, "data", "adversarial_suites.json")
    baseline_p = os.path.join(base_dir, "data", "baseline_results.json")
    plot_p = os.path.join(base_dir, "calibration_curve.png")
    report_p = os.path.join(base_dir, "EVALUATION_REPORT.md")

    print("=" * 70)
    print("  HEVA AI - ADVERSARIAL EVALUATION HARNESS")
    print("  Determinism: seed=42, temp=0.0 | Pure Python Architecture")
    print("=" * 70)

    # 1. Ensure ground truth dataset exists
    if not os.path.exists(dataset_p) or not os.path.exists(corpus_p):
        print("\n[Step 1/5] Scaffolding ground-truth dataset and knowledge corpus...")
        scaffold_dataset()
    else:
        print("\n[Step 1/5] Verified ground-truth dataset and knowledge corpus present.")

    # 2. Build adversarial suites
    print("\n[Step 2/5] Generating 4 adversarial test suites (paraphrase, distractor, subtle error, edge case)...")
    suites = build_and_save_adversarial_suites(dataset_p, suites_p, seed=seed)

    # 3. Initialize evaluation harness & target system
    print("\n[Step 3/5] Initializing Target QA RAG System (FAISS) & Evaluation Harness...")
    harness = EvaluationHarness(
        corpus_path=corpus_p,
        seed=seed,
        enable_query_rewriter=not no_rewriter
    )

    # 4. Run evaluation
    total_evals = sum(len(cases) for cases in suites.values())
    print(f"\n[Step 4/5] Executing evaluation across all 5 test suites ({total_evals} total evaluations)...")
    eval_output = harness.run_full_evaluation(suites, plot_path=plot_p)

    # Save baseline snapshot
    import json
    with open(baseline_p, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2)
    print(f"[*] Baseline snapshot saved to {baseline_p}")

    # 5. Generate Markdown report
    print("\n[Step 5/5] Generating comprehensive evaluation report...")
    generate_markdown_report(eval_output, report_p)

    print("\n" + "=" * 70)
    print("EVALUATION HARNESS RUN COMPLETED SUCCESSFULLY!")
    print(f"  - Overall Accuracy: {eval_output['overall_accuracy']:.1%}")
    print(f"  - Overall Hallucination Rate: {eval_output['overall_hallucination_rate']:.1%}")
    print(f"  - Expected Calibration Error (ECE): {eval_output['calibration']['ece']:.3f}")
    print(f"  - Reliability Diagram: {plot_p}")
    print(f"  - Full Report: {report_p}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Adversarial Evaluation Pipeline")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument("--no-rewriter", action="store_true", help="Ablation: Disable contextual query rewriter")
    args = parser.parse_args()

    run_full_pipeline(seed=args.seed, no_rewriter=args.no_rewriter)
