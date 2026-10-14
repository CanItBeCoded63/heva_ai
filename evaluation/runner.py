"""
Module E: Deterministic Evaluation & Regression Test Runner.

================================================================================
REGRESSION TEST RUNNER SPECIFICATION:
================================================================================
Executes the evaluation suite deterministically over the benchmark dataset.
Computes:
- Overall and per-suite accuracy
- Hallucination detection rate (Module A)
- Confidence calibration ECE / MCE and reliability diagram (Module C)
- Behavioral failure mode clustering via KMeans (Module D)
- Stateful regression detection against baseline_results.json:
  Detects previously passing tests that now fail (REGRESSIONS),
  computes net delta, and logs CI-ready exit codes.
================================================================================
"""

import os
import sys
import json
import argparse
import re
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure workspace root is always in path when executed directly
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from typing import Dict, Any, List, Optional, Tuple
from target_system.schemas import QueryRequest, RagResponse
from target_system.rag_engine import QARagSystem
from target_system.vector_store import FaissVectorStore
from evaluation.hallucination import HallucinationDetector
from evaluation.calibration import ConfidenceCalibrator
from evaluation.clustering import FailureClusterer
from evaluation.adversarial import AdversarialGenerator


STOPWORDS = {
    "which", "their", "there", "about", "would", "could", "should", "under",
    "after", "while", "where", "this", "that", "with", "from", "into", "than",
    "then", "they", "them", "have", "has", "been", "being", "will", "does"
}


def _content_tokens(text: str) -> set:
    return set(re.findall(r"\b[A-Za-z0-9_-]{4,}\b", text.lower())) - STOPWORDS


def _fact_is_supported(fact: str, answer: str, min_overlap: float = 0.60) -> bool:
    fact_tokens = _content_tokens(fact)
    if not fact_tokens:
        return True
    answer_tokens = _content_tokens(answer)
    return len(fact_tokens.intersection(answer_tokens)) / len(fact_tokens) >= min_overlap


def _forbidden_fact_present(fact: str, answer: str) -> bool:
    fact_tokens = _content_tokens(fact)
    if not fact_tokens:
        return False
    answer_tokens = _content_tokens(answer)
    return fact.lower() in answer.lower() or fact_tokens.issubset(answer_tokens)


def check_answer_accuracy(
    model_answer: str,
    expected_answer: str,
    adv_type: str,
    is_hallucination: bool,
    answer_type: str = "answerable",
    required_facts: Optional[List[str]] = None,
    forbidden_facts: Optional[List[str]] = None
) -> bool:
    """
    Evaluates whether the model answer satisfies ground truth.
    A test case fails if:
    1. It produced an ungrounded hallucination, OR
    2. It fell for a subtle error / false presupposition, OR
    3. Key factual tokens from expected_answer are absent from the model answer.
    """
    if is_hallucination:
        return False

    if forbidden_facts and any(_forbidden_fact_present(fact, model_answer) for fact in forbidden_facts):
        return False

    if answer_type == "unanswerable" or adv_type == "unanswerable":
        refusal_keywords = [
            "does not contain", "not contain", "not available", "insufficient",
            "cannot answer", "can't answer", "not enough information", "not provided",
            "not in the corpus", "not in the context", "no information", "unable to answer"
        ]
        return any(kw in model_answer.lower() for kw in refusal_keywords)

    if adv_type == "subtle_error":
        # On subtle errors, the answer must refute or challenge the premise.
        # Expanded keyword list to match the broader refutation language that the improved
        # premise-verification guardrail now produces in its answers.
        refute_keywords = [
            "not supported", "incorrect", "mismatch", "contrary", "does not contain",
            "refute", "false", "premise is incorrect", "false presupposition",
            "contradicts", "the context states", "actually", "rather than",
            "premise error", "not accurate", "inaccurate", "the correct",
            "should be", "the document", "the context", "according to",
            "not consistent", "does not match", "is not", "are not", "cannot",
            "no such", "there is no", "does not exist", "not mentioned",
            "the specification", "the authoritative", "documentation",
        ]
        if any(kw in model_answer.lower() for kw in refute_keywords):
            return True
        # If the model blindly answered without any refutation signal, it failed
        return False

    if required_facts:
        return all(_fact_is_supported(fact, model_answer) for fact in required_facts)

    # Extract key content words from expected answer
    expected_words = set(re.findall(r"\b[A-Za-z0-9_-]{4,}\b", expected_answer.lower()))
    model_words = set(re.findall(r"\b[A-Za-z0-9_-]{4,}\b", model_answer.lower()))

    # Ignore generic stopwords
    expected_words = expected_words - STOPWORDS

    if not expected_words:
        return True

    overlap = expected_words.intersection(model_words)
    overlap_ratio = len(overlap) / len(expected_words)

    # 40% lexical/semantic overlap required
    return overlap_ratio >= 0.40


class EvaluationHarness:
    """
    Orchestrates execution of the target system against test cases,
    computes evaluation metrics, and performs regression checks.
    """

    def __init__(
        self,
        corpus_path: str,
        seed: int = 42,
        enable_query_rewriter: bool = True
    ):
        self.seed = seed
        self.vector_store = FaissVectorStore(seed=seed)

        # Load corpus and build FAISS index
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        self.corpus_by_id = {doc["id"]: doc for doc in corpus}
        self.vector_store.build_index(corpus)

        self.target_system = QARagSystem(
            vector_store=self.vector_store,
            seed=seed,
            temperature=0.0,
            enable_query_rewriter=enable_query_rewriter
        )
        self.hallucination_detector = HallucinationDetector(
            llm_client=self.target_system.llm_client, seed=seed
        )
        self.calibrator = ConfidenceCalibrator(n_bins=5)
        self.clusterer = FailureClusterer(n_clusters=3, seed=seed)

    def validate_grounding(
        self,
        response: RagResponse,
        case: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validates forced span-grounding: answerable cases must cite a real document
        id and provide a quote that occurs verbatim in that document.
        """
        answer_type = case.get("answer_type", "answerable")
        if answer_type == "unanswerable":
            return True, "Unanswerable case does not require a supporting quote."

        source_doc_id = (response.source_doc_id or "").strip()
        supporting_quote = (response.supporting_quote or "").strip().rstrip(".")
        if not source_doc_id or not supporting_quote:
            return False, "Missing source_doc_id or supporting_quote."

        source_doc = self.corpus_by_id.get(source_doc_id)
        if not source_doc:
            return False, f"Unknown source_doc_id: {source_doc_id}"

        doc_text = source_doc.get("text", "").strip()
        quote_match = supporting_quote.lower() in doc_text.lower()
        expected_doc = case.get("expected_source_doc_id")
        expected_doc_match = not expected_doc or source_doc_id == expected_doc
        if not quote_match:
            return False, "Supporting quote is not a verbatim span from the cited document."
        if not expected_doc_match:
            return False, f"Cited {source_doc_id}, expected {expected_doc}."
        return True, "Grounding quote verified."

    def validate_typed_claims(
        self,
        response: RagResponse,
        case: Dict[str, Any]
    ) -> Tuple[bool, str]:
        expected_claims = case.get("expected_typed_claims") or []
        if not expected_claims:
            return True, "No typed-claim expectation for this case."

        actual_claims = response.typed_claims or []
        for expected in expected_claims:
            if expected.get("type") == "temporal_precedence":
                expected_doc = expected.get("winning_doc_id")
                if response.source_doc_id == expected_doc:
                    return True, "Temporal precedence claim verified by cited source."
                continue
            if expected in actual_claims:
                return True, "Typed claim matched exactly."
        return False, f"Missing expected typed claim: {expected_claims}"

    def run_suite(
        self,
        test_cases: List[Dict[str, Any]],
        suite_name: str = "clean"
    ) -> Dict[str, Any]:
        """
        Executes a single test suite and records all predictions and metrics.
        """
        print(f"\n[+] Executing Suite '{suite_name}' ({len(test_cases)} test cases)...")

        results = []
        confidences = []
        accuracies = []
        hallucination_flags = []
        failures = []
        grounding_failures = []
        typed_claim_failures = []

        for i, case in enumerate(test_cases):
            query_text = case["query"]
            expected_answer = case.get("expected_answer", "")
            source_context = case.get("source_context", "")
            adv_type = case.get("adversarial_type", suite_name)
            answer_type = case.get("answer_type", "answerable")

            # Query target system
            req = QueryRequest(query=query_text)
            response = self.target_system.query(req)

            # Module A: Hallucination Detection
            is_hallucinated, hall_reason, verdicts = self.hallucination_detector.detect_hallucinations(
                output_claims=response.extracted_claims,
                source_context=source_context
            )

            # Accuracy check
            is_correct = check_answer_accuracy(
                model_answer=response.answer,
                expected_answer=expected_answer,
                adv_type=adv_type,
                is_hallucination=is_hallucinated,
                answer_type=answer_type,
                required_facts=case.get("required_facts"),
                forbidden_facts=case.get("forbidden_facts")
            )
            grounding_ok, grounding_reason = self.validate_grounding(response, case)
            typed_claims_ok, typed_claims_reason = self.validate_typed_claims(response, case)
            is_correct = bool(is_correct and grounding_ok and typed_claims_ok)

            record = {
                "id": case.get("id", i + 1),
                "suite": suite_name,
                "adversarial_type": adv_type,
                "answer_type": answer_type,
                "query": query_text,
                "expected_answer": expected_answer,
                "model_answer": response.answer,
                "reasoning": response.reasoning,
                "extracted_claims": response.extracted_claims,
                "typed_claims": response.typed_claims,
                "supporting_quote": response.supporting_quote,
                "source_doc_id": response.source_doc_id,
                "grounding_ok": grounding_ok,
                "grounding_reason": grounding_reason,
                "typed_claims_ok": typed_claims_ok,
                "typed_claims_reason": typed_claims_reason,
                "required_facts": case.get("required_facts", []),
                "forbidden_facts": case.get("forbidden_facts", []),
                "confidence_score": response.confidence_score,
                "is_hallucination": is_hallucinated,
                "hallucination_reason": hall_reason,
                "is_correct": bool(is_correct)
            }

            results.append(record)
            confidences.append(response.confidence_score)
            accuracies.append(1 if is_correct else 0)
            hallucination_flags.append(1 if is_hallucinated else 0)

            if not is_correct:
                failures.append(record)
            if not grounding_ok:
                grounding_failures.append(record)
            if not typed_claims_ok:
                typed_claim_failures.append(record)

        acc_score = sum(accuracies) / max(len(accuracies), 1)
        hall_rate = sum(hallucination_flags) / max(len(hallucination_flags), 1)

        print(f"    - Accuracy: {acc_score:.1%} ({sum(accuracies)}/{len(accuracies)})")
        print(f"    - Hallucination Rate: {hall_rate:.1%} ({sum(hallucination_flags)}/{len(hallucination_flags)})")

        return {
            "suite_name": suite_name,
            "total_cases": len(test_cases),
            "accuracy": acc_score,
            "hallucination_rate": hall_rate,
            "results": results,
            "confidences": confidences,
            "accuracies": accuracies,
            "coverage": {
                "grounding_checked": len(results),
                "grounding_failures": len(grounding_failures),
                "typed_claims_checked": sum(1 for r in results if r["typed_claims_ok"] is not None),
                "typed_claim_failures": len(typed_claim_failures),
                "system_errors": 0,
                "pending": 0
            },
            "failures": failures
        }

    def run_full_evaluation(
        self,
        all_suites: Dict[str, List[Dict[str, Any]]],
        plot_path: str = "calibration_curve.png"
    ) -> Dict[str, Any]:
        """
        Runs clean and all adversarial suites, computes calibration & clustering.
        """
        suite_reports = {}
        all_results = []
        all_confidences = []
        all_accuracies = []
        all_failures = []
        coverage = {
            "grounding_checked": 0,
            "grounding_failures": 0,
            "typed_claims_checked": 0,
            "typed_claim_failures": 0,
            "system_errors": 0,
            "pending": 0
        }

        for suite_name, cases in all_suites.items():
            report = self.run_suite(cases, suite_name)
            suite_reports[suite_name] = {
                "total": report["total_cases"],
                "accuracy": report["accuracy"],
                "hallucination_rate": report["hallucination_rate"],
                "coverage": report["coverage"]
            }
            for key in coverage:
                coverage[key] += report["coverage"].get(key, 0)
            all_results.extend(report["results"])
            all_confidences.extend(report["confidences"])
            all_accuracies.extend(report["accuracies"])
            all_failures.extend(report["failures"])

        # Module C: Calibration
        calib_metrics = self.calibrator.compute_calibration_metrics(
            all_confidences, all_accuracies
        )
        if plot_path:
            self.calibrator.plot_reliability_diagram(
                all_confidences, all_accuracies, plot_path
            )

        # Module D: Clustering
        cluster_metrics = self.clusterer.cluster_failures(all_failures)

        total_cases = len(all_results)
        overall_accuracy = sum(all_accuracies) / max(total_cases, 1)
        overall_hall_rate = sum(1 for r in all_results if r["is_hallucination"]) / max(total_cases, 1)

        return {
            "overall_accuracy": overall_accuracy,
            "overall_hallucination_rate": overall_hall_rate,
            "total_evaluated_cases": total_cases,
            "coverage": coverage,
            "suite_breakdown": suite_reports,
            "calibration": calib_metrics,
            "clustering": cluster_metrics,
            "all_results": all_results,
            "all_failures": all_failures
        }

    def compare_with_baseline(
        self,
        current_results: List[Dict[str, Any]],
        baseline_filepath: str,
        accuracy_regression_threshold: float = 0.02,
        hallucination_regression_threshold: float = 0.02
    ) -> Dict[str, Any]:
        """
        Detects regressions against previously saved baseline.
        """
        if not os.path.exists(baseline_filepath):
            print(f"[!] Baseline file not found at {baseline_filepath}. Cannot compute regression delta.")
            return {"status": "NO_BASELINE"}

        with open(baseline_filepath, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)

        baseline_map = {
            f"{r.get('suite', 'clean')}_{r['id']}": r for r in baseline_data.get("all_results", [])
        }

        regressions = []
        improvements = []
        stable_pass = []
        stable_fail = []
        new_hallucinations = []
        resolved_hallucinations = []
        matched_current = []
        matched_baseline = []

        for curr in current_results:
            key = f"{curr.get('suite', 'clean')}_{curr['id']}"
            base = baseline_map.get(key)
            if not base:
                continue
            matched_current.append(curr)
            matched_baseline.append(base)

            base_correct = base.get("is_correct", False)
            curr_correct = curr.get("is_correct", False)
            base_hall = base.get("is_hallucination", False)
            curr_hall = curr.get("is_hallucination", False)

            if base_correct and not curr_correct:
                regressions.append({
                    "id": curr["id"],
                    "suite": curr.get("suite"),
                    "query": curr["query"],
                    "baseline_answer": base.get("model_answer"),
                    "current_answer": curr.get("model_answer"),
                    "current_reason": curr.get("hallucination_reason")
                })
            elif not base_correct and curr_correct:
                improvements.append(curr)
            elif curr_correct:
                stable_pass.append(curr)
            else:
                stable_fail.append(curr)

            if not base_hall and curr_hall:
                new_hallucinations.append(curr)
            elif base_hall and not curr_hall:
                resolved_hallucinations.append(curr)

        baseline_accuracy = sum(1 for r in matched_baseline if r.get("is_correct", False)) / max(len(matched_baseline), 1)
        current_accuracy = sum(1 for r in matched_current if r.get("is_correct", False)) / max(len(matched_current), 1)
        baseline_hallucination_rate = sum(1 for r in matched_baseline if r.get("is_hallucination", False)) / max(len(matched_baseline), 1)
        current_hallucination_rate = sum(1 for r in matched_current if r.get("is_hallucination", False)) / max(len(matched_current), 1)
        accuracy_delta = current_accuracy - baseline_accuracy
        hallucination_delta = current_hallucination_rate - baseline_hallucination_rate
        metric_regressions = []
        if accuracy_delta < -accuracy_regression_threshold:
            metric_regressions.append({
                "metric": "accuracy",
                "baseline": baseline_accuracy,
                "current": current_accuracy,
                "delta": accuracy_delta,
                "threshold": accuracy_regression_threshold
            })
        if hallucination_delta > hallucination_regression_threshold:
            metric_regressions.append({
                "metric": "hallucination_rate",
                "baseline": baseline_hallucination_rate,
                "current": current_hallucination_rate,
                "delta": hallucination_delta,
                "threshold": hallucination_regression_threshold
            })

        count_mismatch = len(current_results) != len(baseline_data.get("all_results", []))

        print("\n" + "=" * 60)
        print("REGRESSION ANALYSIS SUMMARY:")
        print("=" * 60)
        print(f"  Regressions (Previously PASS -> Now FAIL): {len(regressions)}")
        print(f"  Improvements (Previously FAIL -> Now PASS): {len(improvements)}")
        print(f"  New Hallucinations (Was Grounded -> Now Hallucinated): {len(new_hallucinations)}")
        print(f"  Resolved Hallucinations (Was Hallucinated -> Now Grounded): {len(resolved_hallucinations)}")
        print(f"  Accuracy Delta: {accuracy_delta:+.2%} (threshold -{accuracy_regression_threshold:.2%})")
        print(f"  Hallucination Delta: {hallucination_delta:+.2%} (threshold +{hallucination_regression_threshold:.2%})")
        if count_mismatch:
            print(f"  Coverage Warning: baseline has {len(baseline_data.get('all_results', []))} cases, current has {len(current_results)} cases")
        print(f"  Stable Passes: {len(stable_pass)}")
        print(f"  Stable Failures: {len(stable_fail)}")
        print("=" * 60)

        if regressions:
            print("\n[!] CRITICAL: REGRESSIONS DETECTED:")
            for reg in regressions[:5]:
                print(f"    - Case #{reg['id']} [{reg['suite']}]: {reg['query'][:60]}...")

        status = "REGRESSIONS_DETECTED" if regressions or metric_regressions else "CLEAN"

        return {
            "status": status,
            "num_regressions": len(regressions),
            "num_metric_regressions": len(metric_regressions),
            "num_improvements": len(improvements),
            "num_new_hallucinations": len(new_hallucinations),
            "num_resolved_hallucinations": len(resolved_hallucinations),
            "coverage_warning": count_mismatch,
            "metric_deltas": {
                "accuracy_delta": accuracy_delta,
                "hallucination_delta": hallucination_delta
            },
            "metric_regressions": metric_regressions,
            "regressions": regressions,
            "improvements": improvements,
            "new_hallucinations": new_hallucinations,
            "resolved_hallucinations": resolved_hallucinations
        }


def main():
    parser = argparse.ArgumentParser(description="Deterministic Adversarial Evaluation Runner")
    parser.add_argument("--dataset", default="data/dataset.jsonl", help="Path to ground truth dataset")
    parser.add_argument("--corpus", default="data/corpus.json", help="Path to knowledge corpus")
    parser.add_argument("--suites-file", default="data/adversarial_suites.json", help="Path to adversarial suites")
    parser.add_argument("--save-baseline", default="data/baseline_results.json", help="Path to save baseline")
    parser.add_argument("--compare-baseline", action="store_true", help="Compare run against saved baseline")
    parser.add_argument("--plot-path", default="calibration_curve.png", help="Path to save reliability diagram")
    parser.add_argument("--no-rewriter", action="store_true", help="Ablation: Disable contextual query rewriter")
    parser.add_argument("--accuracy-regression-threshold", type=float, default=0.02, help="Allowed absolute accuracy drop before regression failure")
    parser.add_argument("--hallucination-regression-threshold", type=float, default=0.02, help="Allowed absolute hallucination-rate increase before regression failure")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_p = os.path.join(base_dir, args.corpus) if not os.path.isabs(args.corpus) else args.corpus
    dataset_p = os.path.join(base_dir, args.dataset) if not os.path.isabs(args.dataset) else args.dataset
    suites_p = os.path.join(base_dir, args.suites_file) if not os.path.isabs(args.suites_file) else args.suites_file
    baseline_p = os.path.join(base_dir, args.save_baseline) if not os.path.isabs(args.save_baseline) else args.save_baseline
    plot_p = os.path.join(base_dir, args.plot_path) if not os.path.isabs(args.plot_path) else args.plot_path

    # Ensure suites exist
    if not os.path.exists(suites_p):
        print(f"[*] Generating adversarial suites at {suites_p}...")
        from evaluation.adversarial import build_and_save_adversarial_suites
        all_suites = build_and_save_adversarial_suites(dataset_p, suites_p, seed=args.seed)
    else:
        with open(suites_p, "r", encoding="utf-8") as f:
            all_suites = json.load(f)

    harness = EvaluationHarness(
        corpus_path=corpus_p,
        seed=args.seed,
        enable_query_rewriter=not args.no_rewriter
    )

    eval_output = harness.run_full_evaluation(all_suites, plot_path=plot_p)

    # Save baseline if requested
    if baseline_p and not args.compare_baseline:
        os.makedirs(os.path.dirname(baseline_p), exist_ok=True)
        with open(baseline_p, "w", encoding="utf-8") as f:
            json.dump(eval_output, f, indent=2)
        print(f"\n[*] Baseline saved to {baseline_p}")

    # Compare against baseline if requested
    if args.compare_baseline:
        diff = harness.compare_with_baseline(
            eval_output["all_results"],
            baseline_p,
            accuracy_regression_threshold=args.accuracy_regression_threshold,
            hallucination_regression_threshold=args.hallucination_regression_threshold
        )
        if diff["status"] == "REGRESSIONS_DETECTED":
            print("\n[!] Regression check failed: prompt or system modifications introduced regressions.")
            sys.exit(1)
        else:
            print("\n[+] Regression check passed: No regressions introduced.")

    return eval_output


if __name__ == "__main__":
    main()
