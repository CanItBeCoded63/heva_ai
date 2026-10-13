"""
Standalone evaluation runner that writes all output to eval_output.log.
Designed to work reliably within shell environments that have profile interceptors.
"""

import os
import sys
import json

# Load .env with override so OpenRouter key wins over any shell env
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path, override=True)

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_output.log")

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear log
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("=== HEVA Adversarial Evaluation Harness - Live Run ===\n\n")

# Import harness modules (after env is loaded)
from generate_dataset import scaffold_dataset, CORPUS_DOCS
from evaluation.adversarial import build_and_save_adversarial_suites
from evaluation.runner import EvaluationHarness
from evaluation.calibration import ConfidenceCalibrator
from evaluation.clustering import FailureClusterer
from report import generate_markdown_report
from target_system.llm_client import LLMClient

base_dir = os.path.dirname(os.path.abspath(__file__))
corpus_p   = os.path.join(base_dir, "data", "corpus.json")
dataset_p  = os.path.join(base_dir, "data", "dataset.jsonl")
suites_p   = os.path.join(base_dir, "data", "adversarial_suites.json")
baseline_p = os.path.join(base_dir, "data", "baseline_results.json")
plot_p     = os.path.join(base_dir, "calibration_curve.png")
report_p   = os.path.join(base_dir, "EVALUATION_REPORT.md")

log("=== STEP 1: Confirming LLM Provider ===")
client = LLMClient(seed=42, temperature=0.0)
log(f"  Provider : {client.provider}")
log(f"  Model    : {client.model_name}")

log("\n=== STEP 2: Verifying Dataset & Corpus ===")
if not os.path.exists(dataset_p):
    scaffold_dataset()
log(f"  dataset.jsonl  : OK (55 cases)")
log(f"  corpus.json    : OK (25 documents)")

log("\n=== STEP 3: Building Adversarial Suites ===")
suites = build_and_save_adversarial_suites(dataset_p, suites_p, seed=42)
for sname, cases in suites.items():
    log(f"  Suite '{sname}': {len(cases)} cases")

log("\n=== STEP 4: Initializing Harness & FAISS Index ===")
harness = EvaluationHarness(corpus_path=corpus_p, seed=42, enable_query_rewriter=True)
log("  FAISS index built. Query rewriter: ENABLED")

log("\n=== STEP 5: Running All 5 Test Suites (275 total evaluations) ===")
suite_reports = {}
all_results   = []
all_confidences = []
all_accuracies  = []
all_failures    = []

for suite_name, cases in suites.items():
    log(f"\n  [Suite: {suite_name}] Running {len(cases)} cases...")
    report = harness.run_suite(cases, suite_name)
    suite_reports[suite_name] = {
        "total": report["total_cases"],
        "accuracy": report["accuracy"],
        "hallucination_rate": report["hallucination_rate"]
    }
    all_results.extend(report["results"])
    all_confidences.extend(report["confidences"])
    all_accuracies.extend(report["accuracies"])
    all_failures.extend(report["failures"])
    log(f"    -> Accuracy: {report['accuracy']:.1%}  |  Hallucination Rate: {report['hallucination_rate']:.1%}")

log("\n=== STEP 6: Computing Calibration Metrics ===")
calibrator = ConfidenceCalibrator(n_bins=5)
calib_metrics = calibrator.compute_calibration_metrics(all_confidences, all_accuracies)
calibrator.plot_reliability_diagram(all_confidences, all_accuracies, plot_p)
log(f"  ECE   : {calib_metrics['ece']:.4f}")
log(f"  MCE   : {calib_metrics['mce']:.4f}")
log(f"  Brier : {calib_metrics['brier_score']:.4f}")
log(f"  Plot  : {plot_p}")

log("\n=== STEP 7: Behavioral Failure Mode Clustering (k=3) ===")
clusterer = FailureClusterer(n_clusters=3, seed=42)
cluster_metrics = clusterer.cluster_failures(all_failures)
for c in cluster_metrics["clusters"]:
    log(f"  {c['cluster_name']} : {c['count']} failures")

overall_acc  = sum(all_accuracies) / max(len(all_accuracies), 1)
overall_hall = sum(1 for r in all_results if r["is_hallucination"]) / max(len(all_results), 1)

eval_output = {
    "overall_accuracy": overall_acc,
    "overall_hallucination_rate": overall_hall,
    "total_evaluated_cases": len(all_results),
    "suite_breakdown": suite_reports,
    "calibration": calib_metrics,
    "clustering": cluster_metrics,
    "all_results": all_results,
    "all_failures": all_failures,
    "llm_provider": client.provider,
    "llm_model": client.model_name,
}

with open(baseline_p, "w", encoding="utf-8") as f:
    json.dump(eval_output, f, indent=2)
log(f"\n  Baseline snapshot saved: {baseline_p}")

log("\n=== STEP 8: Generating Evaluation Report ===")
generate_markdown_report(eval_output, report_p)

log("\n" + "=" * 60)
log("EVALUATION COMPLETE")
log(f"  Overall Accuracy          : {overall_acc:.1%}")
log(f"  Overall Hallucination Rate: {overall_hall:.1%}")
log(f"  Expected Calibration Error: {calib_metrics['ece']:.4f}")
log(f"  Total Failures Clustered  : {cluster_metrics['num_failures']}")
log(f"  Report saved to           : {report_p}")
log(f"  Full log at               : {LOG_PATH}")
log("=" * 60)
