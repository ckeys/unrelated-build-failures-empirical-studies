#!/usr/bin/env python3
"""
PU Learning Validation for Keycloak Dataset.

This script validates the PU Learning approach using the labeled P, Q, N datasets.
It uses the same methodology as the original JIRA study.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import pulearn
try:
    from pulearn import ElkanotoPuClassifier, WeightedElkanotoPuClassifier
    PULEARN_AVAILABLE = True
except ImportError:
    PULEARN_AVAILABLE = False
    print("Warning: pulearn not available. Install with: pip install pulearn")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rebuttal_data" / "github"

# Configuration
RANDOM_STATE = 42
N_SPLITS = 10
N_ESTIMATORS = 100

# Features to use (same as original study)
FEATURE_COLUMNS = [
    'num_parallel_issues',
    'num_prior_comments',
    'ci_latency_days',
    'has_code_patch',
    'is_shared_same_emsg',
    'num_similar_failures',
    'has_config_files',
    'config_files_count',
    'config_lines_added',
    'config_lines_deleted',
    'config_lines_modified',
    'has_source_code',
    'source_code_files_count',
    'source_code_lines_added',
    'source_code_lines_deleted',
    'source_code_lines_modified',
]


def load_datasets(p_ratio: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load P, Q, N datasets.
    
    Args:
        p_ratio: Ratio of P to total positive (P+Q). Default 0.5 following original study.
    
    Returns:
        Tuple of (P, Q, N) DataFrames.
    """
    p_df = pd.read_csv(DATA_DIR / "keycloak_p_dataset.csv")
    q_df = pd.read_csv(DATA_DIR / "keycloak_q_dataset.csv")
    n_df = pd.read_csv(DATA_DIR / "keycloak_n_dataset.csv")
    
    # Adjust P size to be p_ratio of total positive
    total_positive = len(p_df) + len(q_df)
    target_p_size = int(total_positive * p_ratio)
    
    if target_p_size > len(p_df):
        # Need to move some Q samples to P
        additional_needed = target_p_size - len(p_df)
        
        # Randomly select from Q to add to P
        np.random.seed(RANDOM_STATE)
        q_to_p_indices = np.random.choice(len(q_df), size=additional_needed, replace=False)
        
        q_to_p = q_df.iloc[q_to_p_indices].copy()
        q_remaining = q_df.drop(q_df.index[q_to_p_indices]).copy()
        
        # Combine original P with moved Q samples
        p_df = pd.concat([p_df, q_to_p], ignore_index=True)
        q_df = q_remaining.reset_index(drop=True)
    
    return p_df, q_df, n_df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for modeling."""
    # Select only feature columns that exist
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    
    # Convert boolean columns to int
    for col in available_features:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
    
    # Fill NaN with 0
    df[available_features] = df[available_features].fillna(0)
    
    return df[available_features]


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> dict:
    """Calculate evaluation metrics."""
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    return metrics


def run_pu_learning_validation(
    p_df: pd.DataFrame,
    q_df: pd.DataFrame,
    n_df: pd.DataFrame,
    use_weighted: bool = False
) -> dict:
    """
    Run PU Learning validation following the original JIRA study methodology.
    
    Training setup (PU Learning):
    - P (labeled positive, label=1)
    - Q (unlabeled positive, label=0 in training, but actually positive)
    - N (unlabeled negative, label=0 in training, but actually negative)
    
    Testing: Evaluate on held-out P+Q (positive) vs N (negative)
    """
    print(f"\n{'='*60}")
    print(f"PU Learning Validation ({'Weighted' if use_weighted else 'Standard'})")
    print(f"{'='*60}")
    
    # Prepare features
    p_X = prepare_features(p_df.copy())
    q_X = prepare_features(q_df.copy())
    n_X = prepare_features(n_df.copy())
    
    # Labels for PU learning: P=1, Q=0 (unlabeled), N=0 (unlabeled)
    p_y_train = np.ones(len(p_X))
    q_y_train = np.zeros(len(q_X))  # Unlabeled in training
    n_y_train = np.zeros(len(n_X))  # Unlabeled in training
    
    # True labels for evaluation: P=1, Q=1, N=0
    p_y_true = np.ones(len(p_X))
    q_y_true = np.ones(len(q_X))
    n_y_true = np.zeros(len(n_X))
    
    print(f"P (labeled positive): {len(p_X)}")
    print(f"Q (unlabeled positive): {len(q_X)}")
    print(f"N (unlabeled negative): {len(n_X)}")
    
    all_metrics = []
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # Synchronized K-fold split across P, Q, N
    p_splits = list(kf.split(p_X))
    q_splits = list(kf.split(q_X))
    n_splits = list(kf.split(n_X))
    
    for fold in range(N_SPLITS):
        train_idx_p, test_idx_p = p_splits[fold]
        train_idx_q, test_idx_q = q_splits[fold]
        train_idx_n, test_idx_n = n_splits[fold]
        
        # Training data: combine P, Q, N train splits
        train_X = pd.concat([
            p_X.iloc[train_idx_p],
            q_X.iloc[train_idx_q],
            n_X.iloc[train_idx_n]
        ], ignore_index=True)
        
        train_y = np.concatenate([
            p_y_train[train_idx_p],
            q_y_train[train_idx_q],
            n_y_train[train_idx_n]
        ])
        
        # Test data: P+Q test (positive) vs N test (negative)
        test_pq_X = pd.concat([
            p_X.iloc[test_idx_p],
            q_X.iloc[test_idx_q]
        ], ignore_index=True)
        test_pq_y = np.concatenate([
            p_y_true[test_idx_p],
            q_y_true[test_idx_q]
        ])
        
        test_n_X = n_X.iloc[test_idx_n]
        test_n_y = n_y_true[test_idx_n]
        
        # Skip if test set is too small
        if len(test_pq_y) < 3 or len(test_n_y) < 3:
            continue
        
        # Train PU classifier
        base_estimator = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1
        )
        
        if PULEARN_AVAILABLE:
            if use_weighted:
                labeled_count = len(train_idx_p)
                unlabeled_count = len(train_idx_q) + len(train_idx_n)
                clf = WeightedElkanotoPuClassifier(
                    estimator=base_estimator,
                    labeled=labeled_count,
                    unlabeled=unlabeled_count
                )
            else:
                clf = ElkanotoPuClassifier(
                    estimator=base_estimator,
                    hold_out_ratio=0.1
                )
            
            clf.fit(train_X.values, train_y)
            
            # Predict on P+Q test set
            pq_pred = clf.predict(test_pq_X.values)
            pq_pred = np.where(pq_pred == 1, 1, 0)
            
            # Predict on N test set
            n_pred = clf.predict(test_n_X.values)
            n_pred = np.where(n_pred == 1, 1, 0)
            
            # Get probabilities
            try:
                pq_prob = clf.predict_proba(test_pq_X.values)
                n_prob = clf.predict_proba(test_n_X.values)
            except (AttributeError, IndexError):
                pq_prob = None
                n_prob = None
        else:
            clf = base_estimator
            clf.fit(train_X.values, train_y)
            pq_pred = clf.predict(test_pq_X.values)
            n_pred = clf.predict(test_n_X.values)
            pq_prob = clf.predict_proba(test_pq_X.values)[:, 1]
            n_prob = clf.predict_proba(test_n_X.values)[:, 1]
        
        # Combine predictions and true labels
        all_pred = np.concatenate([pq_pred, n_pred])
        all_true = np.concatenate([test_pq_y, test_n_y])
        
        if pq_prob is not None and n_prob is not None:
            all_prob = np.concatenate([pq_prob, n_prob])
        else:
            all_prob = None
        
        # Evaluate
        metrics = evaluate_model(all_true, all_pred, all_prob)
        all_metrics.append(metrics)
    
    # Calculate mean and std
    results = {}
    for metric in ['precision', 'recall', 'f1', 'auc']:
        values = [m[metric] for m in all_metrics]
        results[f'{metric}_mean'] = np.mean(values)
        results[f'{metric}_std'] = np.std(values)
    
    print(f"\nResults ({N_SPLITS}-fold CV):")
    print(f"  Precision: {results['precision_mean']:.3f} +/- {results['precision_std']:.3f}")
    print(f"  Recall:    {results['recall_mean']:.3f} +/- {results['recall_std']:.3f}")
    print(f"  F1-Score:  {results['f1_mean']:.3f} +/- {results['f1_std']:.3f}")
    print(f"  AUC:       {results['auc_mean']:.3f} +/- {results['auc_std']:.3f}")
    
    return results


def run_hpem_baseline(
    q_df: pd.DataFrame,
    n_df: pd.DataFrame
) -> dict:
    """
    Run HPEM (Heuristic from Prior Error Messages) baseline.
    
    HPEM predicts a build as unrelated if it shares error messages
    with the most recent failed build.
    """
    print(f"\n{'='*60}")
    print("HPEM Baseline (Heuristic from Prior Error Messages)")
    print(f"{'='*60}")
    
    # Combine Q and N for evaluation
    test_df = pd.concat([q_df, n_df], ignore_index=True)
    test_y = np.array([1] * len(q_df) + [0] * len(n_df))
    
    # HPEM: predict 1 if is_shared_same_emsg == True
    y_pred = test_df['is_shared_same_emsg'].astype(int).values
    
    # Evaluate
    metrics = evaluate_model(test_y, y_pred)
    
    print(f"\nResults:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1']:.3f}")
    
    return metrics


def run_random_baseline(
    q_df: pd.DataFrame,
    n_df: pd.DataFrame,
    n_iterations: int = 100
) -> dict:
    """Run random baseline for comparison."""
    print(f"\n{'='*60}")
    print("Random Baseline")
    print(f"{'='*60}")
    
    test_y = np.array([1] * len(q_df) + [0] * len(n_df))
    n_samples = len(test_y)
    
    all_metrics = []
    np.random.seed(RANDOM_STATE)
    
    for _ in range(n_iterations):
        y_pred = np.random.randint(0, 2, n_samples)
        metrics = evaluate_model(test_y, y_pred)
        all_metrics.append(metrics)
    
    results = {}
    for metric in ['precision', 'recall', 'f1']:
        values = [m[metric] for m in all_metrics]
        results[f'{metric}_mean'] = np.mean(values)
        results[f'{metric}_std'] = np.std(values)
    
    print(f"\nResults ({n_iterations} iterations):")
    print(f"  Precision: {results['precision_mean']:.3f} +/- {results['precision_std']:.3f}")
    print(f"  Recall:    {results['recall_mean']:.3f} +/- {results['recall_std']:.3f}")
    print(f"  F1-Score:  {results['f1_mean']:.3f} +/- {results['f1_std']:.3f}")
    
    return results


def run_constant_positive_baseline(
    q_df: pd.DataFrame,
    n_df: pd.DataFrame
) -> dict:
    """
    Run Constant Positive Model baseline.
    Always predicts positive (1) for all samples.
    This is a simple baseline to compare against.
    """
    print(f"\n{'='*60}")
    print("Constant Positive Model Baseline")
    print(f"{'='*60}")
    
    # Test set: Q (positive) + N (negative)
    test_y = np.array([1] * len(q_df) + [0] * len(n_df))
    
    # Always predict positive
    y_pred = np.ones(len(test_y))
    
    # Evaluate
    metrics = evaluate_model(test_y, y_pred)
    
    print(f"\nResults:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1']:.3f}")
    
    return metrics


def run_feature_importance_analysis(
    p_df: pd.DataFrame,
    q_df: pd.DataFrame,
    n_df: pd.DataFrame
) -> dict:
    """
    Analyze feature importance using PU Learning model.
    
    Returns:
        Dictionary of feature names to importance scores.
    """
    print(f"\n{'='*60}")
    print("Feature Importance Analysis")
    print(f"{'='*60}")
    
    # Prepare features
    p_X = prepare_features(p_df.copy())
    q_X = prepare_features(q_df.copy())
    n_X = prepare_features(n_df.copy())
    
    feature_names = list(p_X.columns)
    
    # Combine all data for training
    all_X = pd.concat([p_X, q_X, n_X], ignore_index=True)
    
    # PU Learning labels: P=1, Q=0, N=0
    all_y = np.concatenate([
        np.ones(len(p_X)),
        np.zeros(len(q_X)),
        np.zeros(len(n_X))
    ])
    
    # Train model
    base_estimator = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    if PULEARN_AVAILABLE:
        clf = ElkanotoPuClassifier(
            estimator=base_estimator,
            hold_out_ratio=0.1
        )
        clf.fit(all_X.values, all_y)
        
        # Get feature importance from the underlying estimator
        try:
            importances = clf.estimator.feature_importances_
        except AttributeError:
            # Fallback to base estimator
            base_estimator.fit(all_X.values, all_y)
            importances = base_estimator.feature_importances_
    else:
        base_estimator.fit(all_X.values, all_y)
        importances = base_estimator.feature_importances_
    
    # Create importance dictionary
    importance_dict = dict(zip(feature_names, importances))
    
    # Sort by importance
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("\nFeature Importance Ranking:")
    print("-" * 50)
    for i, (feature, importance) in enumerate(sorted_importance, 1):
        bar = "█" * int(importance * 50)
        print(f"{i:2d}. {feature:<35} {importance:.4f} {bar}")
    
    # Top 5 features
    print("\n" + "=" * 50)
    print("Top 5 Most Important Features:")
    print("=" * 50)
    for i, (feature, importance) in enumerate(sorted_importance[:5], 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    return dict(sorted_importance)


def main() -> None:
    """Main entry point."""
    print("=" * 70)
    print("Keycloak PU Learning Validation")
    print("=" * 70)
    
    # Load datasets
    print("\nLoading datasets...")
    p_df, q_df, n_df = load_datasets()
    
    print(f"P (Heuristic Positive): {len(p_df)}")
    print(f"Q (Labeled Positive):   {len(q_df)}")
    print(f"N (Labeled Negative):   {len(n_df)}")
    
    # Check available features
    available_features = [col for col in FEATURE_COLUMNS if col in p_df.columns]
    print(f"\nAvailable features: {len(available_features)}")
    
    # Run baselines
    random_results = run_random_baseline(q_df, n_df)
    constant_positive_results = run_constant_positive_baseline(q_df, n_df)
    hpem_results = run_hpem_baseline(q_df, n_df)
    
    # Run PU Learning
    if PULEARN_AVAILABLE:
        pu_results = run_pu_learning_validation(p_df, q_df, n_df, use_weighted=False)
        weighted_pu_results = run_pu_learning_validation(p_df, q_df, n_df, use_weighted=True)
    else:
        print("\nSkipping PU Learning (pulearn not installed)")
        pu_results = None
        weighted_pu_results = None
    
    # Feature importance analysis
    feature_importance = run_feature_importance_analysis(p_df, q_df, n_df)
    
    # Summary table
    print("\n" + "=" * 70)
    print("Summary Comparison")
    print("=" * 70)
    print(f"{'Model':<30} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 70)
    
    print(f"{'Random Baseline':<30} "
          f"{random_results['precision_mean']:.3f}+/-{random_results['precision_std']:.3f}  "
          f"{random_results['recall_mean']:.3f}+/-{random_results['recall_std']:.3f}  "
          f"{random_results['f1_mean']:.3f}+/-{random_results['f1_std']:.3f}")
    
    print(f"{'Constant Positive':<30} "
          f"{constant_positive_results['precision']:.3f}            "
          f"{constant_positive_results['recall']:.3f}            "
          f"{constant_positive_results['f1']:.3f}")
    
    print(f"{'HPEM Baseline':<30} "
          f"{hpem_results['precision']:.3f}            "
          f"{hpem_results['recall']:.3f}            "
          f"{hpem_results['f1']:.3f}")
    
    if pu_results:
        print(f"{'PU Learning (Elkanoto)':<30} "
              f"{pu_results['precision_mean']:.3f}+/-{pu_results['precision_std']:.3f}  "
              f"{pu_results['recall_mean']:.3f}+/-{pu_results['recall_std']:.3f}  "
              f"{pu_results['f1_mean']:.3f}+/-{pu_results['f1_std']:.3f}")
    
    if weighted_pu_results:
        print(f"{'PU Learning (Weighted)':<30} "
              f"{weighted_pu_results['precision_mean']:.3f}+/-{weighted_pu_results['precision_std']:.3f}  "
              f"{weighted_pu_results['recall_mean']:.3f}+/-{weighted_pu_results['recall_std']:.3f}  "
              f"{weighted_pu_results['f1_mean']:.3f}+/-{weighted_pu_results['f1_std']:.3f}")


if __name__ == "__main__":
    main()

