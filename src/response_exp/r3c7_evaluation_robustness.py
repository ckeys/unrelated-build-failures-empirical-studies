"""Robustness evaluation to address reviewer comment V2.R3.C7.

This script adds a second evaluation approach to complement the paper's
micro-aggregated confusion matrix (i.e., pooling predictions across folds).

We keep the *exact same* preprocessing and data construction used in the main
training pipeline, and only change how we summarize evaluation results:

1) Micro (pooled) metrics per repeat: pool predictions across folds, then compute metrics.
2) Macro (per-fold) metrics per repeat: compute metrics per fold, then average across folds.
3) Repeated evaluation: repeat K-fold splitting with different random_state values and report
   empirical 95% intervals across repeats.

Outputs (written under data/results/evaluation_robustness_r3c7/):
- per_fold_metrics.csv: fold-level metrics for each repeat
- per_repeat_summary.csv: micro + macro metrics per repeat
- summary_ci_by_project.csv: mean and 95% interval per project for micro/macro metrics
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from pulearn import WeightedElkanotoPuClassifier
from tqdm import tqdm

from src.modeling.training.pu_learning_with_cross_validation_v4 import calculate_metrics
from src.rebuttal_exp.threshold_sensitivity_analysis import prepare_data_exactly_like_training


RANDOM_STATE_BASE: int = 42
N_SPLITS: int = 10
N_REPEATS: int = 30
THRESHOLD: float = 0.5
PROJECTS: list[str] = ["hive", "hadoop", "yarn", "hdfs", "hbase", "ambari", "hdds"]


@dataclass(frozen=True)
class FoldMetrics:
    """Container for metrics computed on a single fold."""

    project: str
    repeat: int
    fold: int
    precision: float
    recall: float
    f1: float
    auc: float
    tp: int
    fp: int
    fn: int
    tn: int


@dataclass(frozen=True)
class RepeatSummary:
    """Container for micro/macro summaries for one repeat."""

    project: str
    repeat: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    micro_auc: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    macro_auc: float


def _confusion_counts(true_labels: np.ndarray, pred_probs: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    """Compute TP/FP/FN/TN given PU labels {1, -1} and probability scores.

    Args:
        true_labels: Array with values in {1, -1}.
        pred_probs: Predicted probabilities (or scores) in [0, 1].
        threshold: Decision threshold.

    Returns:
        (tp, fp, fn, tn)
    """
    pred_labels = np.where(pred_probs > threshold, 1, -1)
    tp = int(np.sum((true_labels == 1) & (pred_labels == 1)))
    fn = int(np.sum((true_labels == 1) & (pred_labels == -1)))
    fp = int(np.sum((true_labels == -1) & (pred_labels == 1)))
    tn = int(np.sum((true_labels == -1) & (pred_labels == -1)))
    return tp, fp, fn, tn


def _empirical_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """Compute mean and empirical (percentile) confidence interval.

    Args:
        values: 1D array of metric values across repeats.
        alpha: 1 - confidence level (default 0.05 for 95% interval).

    Returns:
        (mean, lower, upper)
    """
    if values.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    lower = float(np.quantile(values, alpha / 2))
    upper = float(np.quantile(values, 1 - alpha / 2))
    return mean, lower, upper


def evaluate_project_repeated(
    project_name: str,
    base_path: str,
    data_path: str,
    n_repeats: int = N_REPEATS,
    n_splits: int = N_SPLITS,
    threshold: float = THRESHOLD,
    random_state_base: int = RANDOM_STATE_BASE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run repeated K-fold evaluation for a single project.

    Important: Data preparation is done via `prepare_data_exactly_like_training`
    to ensure our robustness evaluation uses the same preprocessing and
    dataset construction as the main pipeline.

    Args:
        project_name: Project name (lowercase).
        base_path: Path to `data/modeling_data/training`.
        data_path: Path to `data/`.
        n_repeats: Number of repeated evaluations (different random_state per repeat).
        n_splits: Number of folds in K-fold.
        threshold: Classification threshold on predicted probabilities.
        random_state_base: Base random seed (repeat i uses base+i).

    Returns:
        (fold_metrics_df, repeat_summary_df)
    """
    P, Q, N, feature_names = prepare_data_exactly_like_training(project_name, base_path, data_path)

    # Prepare features/labels exactly like training
    P_features = P[feature_names]
    Q_features = Q[feature_names]
    N_features = N[feature_names]
    P_labels = P["label"]
    Q_labels = Q["label"]
    N_labels = N["label"]

    fold_rows: list[FoldMetrics] = []
    repeat_rows: list[RepeatSummary] = []

    for repeat in tqdm(range(n_repeats), desc=f"{project_name.upper()} repeats"):
        # Model configuration matches the training script: no estimator random_state set.
        estimator = RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            bootstrap=True,
            n_jobs=-1,
        )
        pu_estimator = WeightedElkanotoPuClassifier(
            estimator=estimator,
            labeled=len(P),
            unlabeled=len(Q) + len(N),
        )

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state_base + repeat)

        pooled_true: list[int] = []
        pooled_prob: list[float] = []
        fold_metrics_values: dict[str, list[float]] = {
            "precision": [],
            "recall": [],
            "f1": [],
            "auc": [],
        }

        for fold_idx, ((train_idx_P, test_idx_P), (train_idx_Q, test_idx_Q), (train_idx_N, test_idx_N)) in enumerate(
            zip(kf.split(P_features), kf.split(Q_features), kf.split(N_features))
        ):
            pu_train_data = pd.concat(
                [
                    P_features.iloc[train_idx_P],
                    Q_features.iloc[train_idx_Q],
                    N_features.iloc[train_idx_N],
                ]
            )
            pu_train_labels = pd.concat(
                [
                    P_labels.iloc[train_idx_P],
                    Q_labels.iloc[train_idx_Q],
                    N_labels.iloc[train_idx_N],
                ]
            )

            pu_estimator.fit(pu_train_data.values, pu_train_labels.values)

            P_Q_test_data = pd.concat([P_features.iloc[test_idx_P], Q_features.iloc[test_idx_Q]])
            P_Q_test_labels = pd.concat([P_labels.iloc[test_idx_P], Q_labels.iloc[test_idx_Q]])

            P_Q_test_prob = pu_estimator.predict_proba(P_Q_test_data.values)
            N_test_prob = pu_estimator.predict_proba(N_features.iloc[test_idx_N].values)

            true_fold = np.array(P_Q_test_labels.tolist() + N_labels.iloc[test_idx_N].tolist())
            prob_fold = np.array(P_Q_test_prob.tolist() + N_test_prob.tolist())

            # Fold-level metrics (macro components)
            precision, recall, f1, auc_value = calculate_metrics(true_fold, prob_fold > threshold, prob_fold)
            tp, fp, fn, tn = _confusion_counts(true_fold, prob_fold, threshold)

            fold_rows.append(
                FoldMetrics(
                    project=project_name,
                    repeat=repeat,
                    fold=fold_idx,
                    precision=float(precision),
                    recall=float(recall),
                    f1=float(f1),
                    auc=float(auc_value),
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    tn=tn,
                )
            )

            fold_metrics_values["precision"].append(float(precision))
            fold_metrics_values["recall"].append(float(recall))
            fold_metrics_values["f1"].append(float(f1))
            fold_metrics_values["auc"].append(float(auc_value))

            # Pooled predictions for micro metrics
            pooled_true.extend(true_fold.tolist())
            pooled_prob.extend(prob_fold.tolist())

        pooled_true_arr = np.array(pooled_true)
        pooled_prob_arr = np.array(pooled_prob)

        micro_precision, micro_recall, micro_f1, micro_auc = calculate_metrics(
            pooled_true_arr, pooled_prob_arr > threshold, pooled_prob_arr
        )

        # Macro metrics: average over fold metrics
        macro_precision = float(np.mean(fold_metrics_values["precision"]))
        macro_recall = float(np.mean(fold_metrics_values["recall"]))
        macro_f1 = float(np.mean(fold_metrics_values["f1"]))
        macro_auc = float(np.mean(fold_metrics_values["auc"]))

        repeat_rows.append(
            RepeatSummary(
                project=project_name,
                repeat=repeat,
                micro_precision=float(micro_precision),
                micro_recall=float(micro_recall),
                micro_f1=float(micro_f1),
                micro_auc=float(micro_auc),
                macro_precision=macro_precision,
                macro_recall=macro_recall,
                macro_f1=macro_f1,
                macro_auc=macro_auc,
            )
        )

    fold_df = pd.DataFrame([asdict(r) for r in fold_rows])
    repeat_df = pd.DataFrame([asdict(r) for r in repeat_rows])
    return fold_df, repeat_df


def summarize_with_ci(repeat_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeat-level metrics with mean and empirical 95% intervals.

    Args:
        repeat_df: DataFrame produced by `evaluate_project_repeated` (repeat-level).

    Returns:
        Summary DataFrame with mean/low/high columns for each metric and aggregation type.
    """
    summary_rows: list[dict[str, Any]] = []
    for project, proj_df in repeat_df.groupby("project"):
        row: dict[str, Any] = {"project": project, "n_repeats": int(len(proj_df))}
        for prefix in ("micro", "macro"):
            for metric in ("precision", "recall", "f1", "auc"):
                values = proj_df[f"{prefix}_{metric}"].to_numpy(dtype=float)
                mean, low, high = _empirical_ci(values, alpha=0.05)
                row[f"{prefix}_{metric}_mean"] = mean
                row[f"{prefix}_{metric}_ci_low"] = low
                row[f"{prefix}_{metric}_ci_high"] = high
        summary_rows.append(row)
    return pd.DataFrame(summary_rows).sort_values("project")


def main() -> None:
    """Entry point."""
    project_root = Path(__file__).resolve().parents[2]
    base_path = str(project_root / "data" / "modeling_data" / "training")
    data_path = str(project_root / "data")

    output_dir = project_root / "data" / "results" / "evaluation_robustness_r3c7"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fold_dfs: list[pd.DataFrame] = []
    all_repeat_dfs: list[pd.DataFrame] = []

    for project in PROJECTS:
        fold_df, repeat_df = evaluate_project_repeated(
            project_name=project,
            base_path=base_path,
            data_path=data_path,
            n_repeats=N_REPEATS,
            n_splits=N_SPLITS,
            threshold=THRESHOLD,
            random_state_base=RANDOM_STATE_BASE,
        )
        all_fold_dfs.append(fold_df)
        all_repeat_dfs.append(repeat_df)

    fold_metrics_df = pd.concat(all_fold_dfs, ignore_index=True)
    repeat_summary_df = pd.concat(all_repeat_dfs, ignore_index=True)
    ci_summary_df = summarize_with_ci(repeat_summary_df)

    fold_metrics_df.to_csv(output_dir / "per_fold_metrics.csv", index=False)
    repeat_summary_df.to_csv(output_dir / "per_repeat_summary.csv", index=False)
    ci_summary_df.to_csv(output_dir / "summary_ci_by_project.csv", index=False)

    # A small convenience combined summary across projects (mean of project means)
    overall = {
        "metric": [],
        "micro_mean_of_project_means": [],
        "macro_mean_of_project_means": [],
    }
    for metric in ("precision", "recall", "f1", "auc"):
        overall["metric"].append(metric)
        overall["micro_mean_of_project_means"].append(float(ci_summary_df[f"micro_{metric}_mean"].mean()))
        overall["macro_mean_of_project_means"].append(float(ci_summary_df[f"macro_{metric}_mean"].mean()))
    pd.DataFrame(overall).to_csv(output_dir / "summary_overall_means.csv", index=False)


if __name__ == "__main__":
    main()


