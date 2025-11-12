#!/usr/bin/env python3
"""Compute probe analysis: linear probe and CAA vectors from hidden state features."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

try:
    import umap
except ImportError:
    umap = None

from setup import setup_logging


LOGGER = logging.getLogger("compute_probes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-data-path",
        type=Path,
        required=True,
        help="Path to .npz file containing probe data (features and labels).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/probe_analysis"),
        help="Directory for saving computed probes, vectors, and metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split and dimensionality reduction.",
    )
    parser.add_argument(
        "--dp-alignment",
        action="store_true",
        help="Restrict analysis to tokens that passed DP alignment (requires metadata in probe data).",
    )
    return parser.parse_args()


def fit_linear_probe(features: np.ndarray, labels: np.ndarray, seed: int = 0) -> Dict:
    """
    Train a logistic regression probe and evaluate it.

    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Binary labels (0=wrong, 1=right)
        seed: Random seed

    Returns:
        Dictionary with accuracy and classification report
    """
    if len(np.unique(labels)) < 2:
        return {"warning": "Not enough class diversity for probe."}

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=seed, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(x_train, y_train)

    preds = clf.predict(x_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    return {
        "accuracy": float(acc),
        "report": report,
        "coefficients": clf.coef_.tolist(),
        "intercept": float(clf.intercept_[0]),
    }


def compute_caa_vector(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute the Contrast Activation Addition (CAA) vector.
    
    Args:
        features: Hidden state features [num_samples, hidden_dim]
        labels: Binary labels (0=incorrect, 1=correct)
    
    Returns:
        CAA vector (difference between correct and incorrect means)
    """
    correct_features = features[labels == 1]
    incorrect_features = features[labels == 0]
    
    if len(correct_features) == 0 or len(incorrect_features) == 0:
        raise ValueError("Need both correct and incorrect samples to compute CAA vector")
    
    correct_mean = correct_features.mean(axis=0)
    incorrect_mean = incorrect_features.mean(axis=0)
    
    caa_vector = correct_mean - incorrect_mean
    return caa_vector


def compute_separation_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    caa_vector: np.ndarray,
) -> Dict[str, float]:
    """
    Compute separation metrics for the CAA vector.
    
    Args:
        features: Hidden state features
        labels: Binary labels
        caa_vector: The computed contrast vector
    
    Returns:
        Dictionary with separation metrics
    """
    # Project features onto CAA direction
    caa_unit = caa_vector / (np.linalg.norm(caa_vector) + 1e-8)
    projections = features @ caa_unit
    
    correct_proj = projections[labels == 1]
    incorrect_proj = projections[labels == 0]
    
    # Compute separation statistics
    correct_mean = correct_proj.mean()
    incorrect_mean = incorrect_proj.mean()
    correct_std = correct_proj.std()
    incorrect_std = incorrect_proj.std()
    
    # Cohen's d effect size
    pooled_std = np.sqrt((correct_std**2 + incorrect_std**2) / 2)
    cohens_d = (correct_mean - incorrect_mean) / (pooled_std + 1e-8)
    
    return {
        "correct_mean": float(correct_mean),
        "incorrect_mean": float(incorrect_mean),
        "correct_std": float(correct_std),
        "incorrect_std": float(incorrect_std),
        "mean_separation": float(correct_mean - incorrect_mean),
        "cohens_d": float(cohens_d),
    }


def compute_pca_transform(features: np.ndarray, seed: int = 0) -> Dict:
    """
    Compute PCA transformation and return components and explained variance.
    
    Args:
        features: Feature matrix
        seed: Random seed
    
    Returns:
        Dictionary with PCA results
    """
    pca = PCA(n_components=2, random_state=seed)
    transformed = pca.fit_transform(features)
    
    return {
        "transformed": transformed,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_.tolist(),
    }


def compute_umap_transform(features: np.ndarray, seed: int = 0) -> Dict:
    """
    Compute UMAP transformation.
    
    Args:
        features: Feature matrix
        seed: Random seed
    
    Returns:
        Dictionary with UMAP results
    """
    if umap is None:
        LOGGER.warning("UMAP not installed, skipping UMAP computation")
        return {"warning": "umap-learn not installed"}
    
    reducer = umap.UMAP(n_components=2, random_state=seed)
    embedded = reducer.fit_transform(features)
    
    return {
        "transformed": embedded,
    }


def main() -> None:
    setup_logging()
    args = parse_args()

    # Load probe data
    if not args.probe_data_path.exists():
        LOGGER.error("Probe data file not found: %s", args.probe_data_path)
        return

    LOGGER.info("Loading probe data from %s", args.probe_data_path)
    data = np.load(args.probe_data_path)
    features = data["features"]
    labels = data["labels"]
    layer = int(data["layer"])

    total_samples = features.shape[0]
    LOGGER.info("Loaded %d samples from layer %d", total_samples, layer)

    mask = np.ones(total_samples, dtype=bool)

    valid_mask = None
    if "valid" in data:
        valid_mask = data["valid"].astype(bool)
        LOGGER.info(
            "Validity flag found: %d valid, %d invalid",
            int(valid_mask.sum()),
            int((~valid_mask).sum()),
        )
        mask &= valid_mask
    else:
        LOGGER.info("No validity flag found; treating all samples as valid.")

    dp_used_alignment = data["dp_used_alignment"].astype(bool) if "dp_used_alignment" in data else None
    dp_used_fallback = data["dp_used_fallback"].astype(bool) if "dp_used_fallback" in data else None

    if args.dp_alignment:
        if dp_used_alignment is None:
            LOGGER.warning(
                "--dp-alignment requested, but probe data lacks 'dp_used_alignment'. Skipping DP filtering."
            )
        else:
            mask &= dp_used_alignment
            if dp_used_fallback is not None:
                mask &= ~dp_used_fallback
            LOGGER.info(
                "DP alignment filter applied: %d of %d samples retained",
                int(mask.sum()),
                total_samples,
            )

    if mask.sum() == 0:
        LOGGER.error("No samples remain after filtering; nothing to analyze.")
        return

    features = features[mask]
    labels = labels[mask]
    dp_used_alignment_subset = dp_used_alignment[mask] if dp_used_alignment is not None else None
    dp_used_fallback_subset = dp_used_fallback[mask] if dp_used_fallback is not None else None

    LOGGER.info("Feature dimension: %d", features.shape[1])
    LOGGER.info(
        "Label distribution: %d wrong, %d right",
        int((labels == 0).sum()),
        int((labels == 1).sum()),
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results
    results = {
        "probe_layer": layer,
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "num_wrong": int((labels == 0).sum()),
        "num_right": int((labels == 1).sum()),
    }
    if dp_used_alignment_subset is not None:
        results["dp_alignment"] = {
            "requested": bool(args.dp_alignment),
            "available": True,
            "num_aligned_samples": int(dp_used_alignment_subset.sum()),
            "num_after_filter": int(features.shape[0]),
        }
    elif args.dp_alignment:
        results["dp_alignment"] = {
            "requested": True,
            "available": False,
            "num_aligned_samples": 0,
            "num_after_filter": int(features.shape[0]),
        }

    # Skip linear probe, just visualize
    LOGGER.info("Skipping linear probe training, computing visualizations only...")

    # 1. Compute PCA
    LOGGER.info("Computing PCA transformation...")
    pca_result = compute_pca_transform(features, seed=args.seed)
    pca_transformed = pca_result.pop("transformed")
    results["pca"] = pca_result
    LOGGER.info("PCA explained variance: %.3f, %.3f", *pca_result["explained_variance_ratio"])
    
    # Save PCA data
    pca_data_path = args.output_dir / f"layer{layer}_pca_data.npz"
    pca_payload = {
        "transformed": pca_transformed,
        "labels": labels,
        "explained_variance_ratio": np.array(pca_result["explained_variance_ratio"]),
    }
    if dp_used_alignment_subset is not None:
        pca_payload["dp_used_alignment"] = dp_used_alignment_subset
    if dp_used_fallback_subset is not None:
        pca_payload["dp_used_fallback"] = dp_used_fallback_subset
    np.savez(pca_data_path, **pca_payload)
    LOGGER.info("Saved PCA data to %s", pca_data_path)

    # 5. Compute UMAP
    LOGGER.info("Computing UMAP transformation...")
    umap_result = compute_umap_transform(features, seed=args.seed)
    if "transformed" in umap_result:
        umap_transformed = umap_result.pop("transformed")
        results["umap"] = umap_result
        
        # Save UMAP data
        umap_data_path = args.output_dir / f"layer{layer}_umap_data.npz"
        umap_payload = {
            "transformed": umap_transformed,
            "labels": labels,
        }
        if dp_used_alignment_subset is not None:
            umap_payload["dp_used_alignment"] = dp_used_alignment_subset
        if dp_used_fallback_subset is not None:
            umap_payload["dp_used_fallback"] = dp_used_fallback_subset
        np.savez(umap_data_path, **umap_payload)
        LOGGER.info("Saved UMAP data to %s", umap_data_path)
    else:
        results["umap"] = umap_result

    # Save summary metrics
    metrics_path = args.output_dir / f"layer{layer}_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved metrics to %s", metrics_path)
    
    LOGGER.info("All probe computations complete!")


if __name__ == "__main__":
    main()
