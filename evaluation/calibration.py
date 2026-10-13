"""
Module C: Confidence Calibration and Reliability Diagram.

================================================================================
CALIBRATION METRICS SPECIFICATION:
================================================================================
Compares the model's self-reported confidence_score against true binary accuracy.
Computes:
1. Expected Calibration Error (ECE):
   ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|
2. Maximum Calibration Error (MCE):
   MCE = max_b |acc(B_b) - conf(B_b)|
3. Brier Score: Mean squared error between confidence and binary outcome.
4. Generates and saves a two-panel Reliability Diagram:
   - Top panel: Empirical accuracy vs. Mean predicted confidence with ideal diagonal
   - Bottom panel: Sample frequency distribution across bins
================================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for server/CLI environments
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from sklearn.calibration import calibration_curve


class ConfidenceCalibrator:
    """
    Evaluates confidence calibration and plots reliability diagrams.
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    def compute_calibration_metrics(
        self,
        confidences: List[float],
        accuracies: List[int]
    ) -> Dict[str, Any]:
        """
        Computes ECE, MCE, and Brier score.
        """
        y_true = np.array(accuracies, dtype=np.int32)
        y_prob = np.array(confidences, dtype=np.float64)

        # Clip probabilities to valid [0, 1] range
        y_prob = np.clip(y_prob, 0.0, 1.0)
        n = len(y_true)
        if n == 0:
            return {"ece": 0.0, "mce": 0.0, "brier": 0.0, "bin_stats": []}

        # Calculate Brier score
        brier_score = float(np.mean((y_prob - y_true) ** 2))

        # Uniform binning for ECE
        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        ece = 0.0
        mce = 0.0
        bin_stats = []

        for b in range(self.n_bins):
            mask = bin_indices == b
            bin_size = int(np.sum(mask))
            if bin_size > 0:
                bin_acc = float(np.mean(y_true[mask]))
                bin_conf = float(np.mean(y_prob[mask]))
                gap = abs(bin_acc - bin_conf)
                ece += (bin_size / n) * gap
                mce = max(mce, gap)
                bin_stats.append({
                    "bin": b,
                    "range": (float(bins[b]), float(bins[b + 1])),
                    "size": bin_size,
                    "accuracy": bin_acc,
                    "confidence": bin_conf,
                    "gap": gap
                })

        # sklearn calibration curve
        try:
            prob_true, prob_pred = calibration_curve(
                y_true, y_prob, n_bins=self.n_bins, strategy="uniform"
            )
        except Exception:
            prob_true, prob_pred = np.array([]), np.array([])

        return {
            "ece": float(ece),
            "mce": float(mce),
            "brier_score": brier_score,
            "mean_confidence": float(np.mean(y_prob)),
            "overall_accuracy": float(np.mean(y_true)),
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "bin_stats": bin_stats
        }

    def plot_reliability_diagram(
        self,
        confidences: List[float],
        accuracies: List[int],
        output_filepath: str,
        title_suffix: str = ""
    ) -> str:
        """
        Generates and saves the reliability diagram PNG.
        """
        metrics = self.compute_calibration_metrics(confidences, accuracies)
        y_true = np.array(accuracies)
        y_prob = np.array(confidences)

        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, figsize=(8, 9), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Top panel: Reliability curve
        ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)

        if metrics["prob_pred"] and metrics["prob_true"]:
            ax1.plot(
                metrics["prob_pred"],
                metrics["prob_true"],
                marker="o",
                color="#0066cc",
                linewidth=2,
                label=f"Target System (ECE = {metrics['ece']:.3f})"
            )

        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel("Mean Predicted Confidence", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
        ax1.set_title(
            f"Confidence Calibration Reliability Diagram {title_suffix}\n"
            f"ECE: {metrics['ece']:.3f} | MCE: {metrics['mce']:.3f} | Brier: {metrics['brier_score']:.3f}",
            fontsize=12,
            pad=12
        )
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", frameon=True)

        # Bottom panel: Confidence histogram
        bins = np.linspace(0.0, 1.0, self.n_bins + 1)
        ax2.hist(y_prob, bins=bins, color="#4d94ff", edgecolor="black", alpha=0.8, rwidth=0.85)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_xlabel("Confidence Bins", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.4)

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        plt.savefig(output_filepath, dpi=300)
        plt.close(fig)

        print(f"[*] Calibration plot saved to {output_filepath}")
        return output_filepath


def evaluate_calibration(
    confidences: List[float],
    accuracies: List[int],
    plot_path: str = "calibration_curve.png"
) -> Dict[str, Any]:
    """
    Convenience function computing metrics and generating the calibration plot.
    """
    calibrator = ConfidenceCalibrator(n_bins=5)
    metrics = calibrator.compute_calibration_metrics(confidences, accuracies)
    calibrator.plot_reliability_diagram(confidences, accuracies, plot_path)
    return metrics
