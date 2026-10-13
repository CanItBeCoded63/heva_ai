"""Evaluation Harness package."""
from evaluation.hallucination import HallucinationDetector, detect_hallucinations
from evaluation.adversarial import AdversarialGenerator
from evaluation.calibration import ConfidenceCalibrator, evaluate_calibration
from evaluation.clustering import FailureClusterer
from evaluation.runner import EvaluationHarness

__all__ = [
    "HallucinationDetector",
    "detect_hallucinations",
    "AdversarialGenerator",
    "ConfidenceCalibrator",
    "evaluate_calibration",
    "FailureClusterer",
    "EvaluationHarness"
]
