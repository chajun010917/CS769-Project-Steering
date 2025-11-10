#!/usr/bin/env python3
"""Compare hidden state embeddings with and without steering to check if they're in the same space."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

try:
    import umap
except ImportError:
    umap = None

from setup import (
    configure_hf_caches,
    setup_logging,
)

LOGGER = logging.getLogger("compare_steering_embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hidden-states-dir",
        type=Path,
        default=Path("artifacts/hidden_states"),
        help="Directory containing stored embeddings (sample_id/right_layer{layer}.pt, wrong_layer{layer}.pt, steered_layer{layer}.pt).",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Layer IDs to analyze (e.g., 26 27 28 29 30 31). Default: 26 27 28 29 30 31",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Single layer ID to analyze (deprecated, use --layers instead).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/steering_embedding_comparison"),
        help="Directory for saving comparison plots.",
    )
    parser.add_argument(
        "--pooling-method",
        type=str,
        choices=["last_token", "mean"],
        default="last_token",
        help="Method to pool hidden states: last_token or mean.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_embeddings_from_files(
    hidden_states_dir: Path,
    sample_ids: List[str],
    layer: int,
    embedding_type: str,
    pooling_method: str = "last_token",
) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings from stored .pt files.
    
    Args:
        hidden_states_dir: Directory containing sample_id subdirectories
        sample_ids: List of sample IDs to load
        layer: Layer ID to load
        embedding_type: 'right', 'wrong', or 'steered'
        pooling_method: Method to pool hidden states ('last_token' or 'mean')
    
    Returns:
        Tuple of (embeddings array [n_samples, hidden_dim], list of loaded sample_ids)
    """
    embeddings_list = []
    loaded_sample_ids = []
    
    for sample_id in tqdm(sample_ids, desc=f"Loading {embedding_type} embeddings"):
        sample_dir = hidden_states_dir / sample_id
        embedding_path = sample_dir / f"{embedding_type}_layer{layer}.pt"
        
        if not embedding_path.exists():
            LOGGER.debug("Embedding not found: %s", embedding_path)
            continue
        
        try:
            # Load tensor [seq_len, hidden_dim]
            tensor = torch.load(embedding_path, map_location="cpu")
            
            # Apply pooling
            if pooling_method == "last_token":
                pooled = tensor[-1].numpy()  # [hidden_dim]
            elif pooling_method == "mean":
                pooled = tensor.mean(dim=0).numpy()  # [hidden_dim]
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}")
            
            embeddings_list.append(pooled)
            loaded_sample_ids.append(sample_id)
        except Exception as e:
            LOGGER.warning("Error loading embedding from %s: %s", embedding_path, e)
            continue
    
    if not embeddings_list:
        LOGGER.warning("No embeddings loaded for layer %d, type %s", layer, embedding_type)
        return np.array([]), []
    
    embeddings_array = np.array(embeddings_list)
    LOGGER.info(
        "Loaded %d %s embeddings for layer %d (shape: %s)",
        len(loaded_sample_ids),
        embedding_type,
        layer,
        embeddings_array.shape,
    )
    return embeddings_array, loaded_sample_ids


def plot_embedding_comparison(
    embeddings_correct: np.ndarray,
    embeddings_wrong_no_steering: np.ndarray,
    embeddings_wrong_with_steering: np.ndarray,
    method: str,
    output_path: Path,
    layer: int,
) -> None:
    """
    Plot comparison of embeddings: correct, wrong without steering, and wrong with steering.
    
    Args:
        embeddings_correct: Correct embeddings [n_samples, hidden_dim]
        embeddings_wrong_no_steering: Wrong embeddings without steering [n_samples, hidden_dim]
        embeddings_wrong_with_steering: Wrong embeddings with steering [n_samples, hidden_dim]
        method: Visualization method ('pca' or 'umap')
        output_path: Path to save plot
        layer: Layer ID
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine embeddings for fitting
    combined_embeddings = np.vstack([
        embeddings_correct,
        embeddings_wrong_no_steering,
        embeddings_wrong_with_steering
    ])
    
    # Apply dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        transformed = reducer.fit_transform(combined_embeddings)
        explained_var = reducer.explained_variance_ratio_
        xlabel = f"PC1 ({explained_var[0]:.1%} variance)"
        ylabel = f"PC2 ({explained_var[1]:.1%} variance)"
    elif method == "umap":
        if umap is None:
            LOGGER.error("UMAP not available. Install umap-learn to use UMAP visualization.")
            return
        reducer = umap.UMAP(n_components=2, random_state=42)
        transformed = reducer.fit_transform(combined_embeddings)
        xlabel = "UMAP Component 1"
        ylabel = "UMAP Component 2"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Split back into correct, wrong no steering, wrong with steering
    n_samples = len(embeddings_correct)
    transformed_correct = transformed[:n_samples]
    transformed_wrong_no_steering = transformed[n_samples:2*n_samples]
    transformed_wrong_with_steering = transformed[2*n_samples:]
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Correct embeddings
    ax1 = axes[0]
    ax1.scatter(
        transformed_correct[:, 0],
        transformed_correct[:, 1],
        alpha=0.6,
        s=30,
        c='green',
        label='Correct',
    )
    ax1.set_title(f'Correct Embeddings (Layer {layer})', fontsize=14, fontweight='bold')
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Wrong without steering
    ax2 = axes[1]
    ax2.scatter(
        transformed_wrong_no_steering[:, 0],
        transformed_wrong_no_steering[:, 1],
        alpha=0.6,
        s=30,
        c='blue',
        label='Wrong (No Steering)',
    )
    ax2.set_title(f'Wrong Embeddings Without Steering (Layer {layer})', fontsize=14, fontweight='bold')
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Wrong with steering
    ax3 = axes[2]
    ax3.scatter(
        transformed_wrong_with_steering[:, 0],
        transformed_wrong_with_steering[:, 1],
        alpha=0.6,
        s=30,
        c='red',
        label='Wrong (With Steering)',
    )
    ax3.set_title(f'Wrong Embeddings With Steering (Layer {layer})', fontsize=14, fontweight='bold')
    ax3.set_xlabel(xlabel, fontsize=12)
    ax3.set_ylabel(ylabel, fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'Embedding Space Comparison: {method.upper()} (Layer {layer})', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    LOGGER.info("Saved comparison plot to %s", output_path)


def plot_overlay_comparison(
    embeddings_correct: np.ndarray,
    embeddings_wrong_no_steering: np.ndarray,
    embeddings_wrong_with_steering: np.ndarray,
    method: str,
    output_path: Path,
    layer: int,
) -> None:
    """
    Plot overlay comparison showing correct, wrong without steering, and wrong with steering in the same space.
    
    Args:
        embeddings_correct: Correct embeddings [n_samples, hidden_dim]
        embeddings_wrong_no_steering: Wrong embeddings without steering [n_samples, hidden_dim]
        embeddings_wrong_with_steering: Wrong embeddings with steering [n_samples, hidden_dim]
        method: Visualization method ('pca' or 'umap')
        output_path: Path to save plot
        layer: Layer ID
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine embeddings for fitting
    combined_embeddings = np.vstack([
        embeddings_correct,
        embeddings_wrong_no_steering,
        embeddings_wrong_with_steering
    ])
    
    # Apply dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        transformed = reducer.fit_transform(combined_embeddings)
        explained_var = reducer.explained_variance_ratio_
        xlabel = f"PC1 ({explained_var[0]:.1%} variance)"
        ylabel = f"PC2 ({explained_var[1]:.1%} variance)"
    elif method == "umap":
        if umap is None:
            LOGGER.error("UMAP not available. Install umap-learn to use UMAP visualization.")
            return
        reducer = umap.UMAP(n_components=2, random_state=42)
        transformed = reducer.fit_transform(combined_embeddings)
        xlabel = "UMAP Component 1"
        ylabel = "UMAP Component 2"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Split back into correct, wrong no steering, wrong with steering
    n_samples = len(embeddings_correct)
    transformed_correct = transformed[:n_samples]
    transformed_wrong_no_steering = transformed[n_samples:2*n_samples]
    transformed_wrong_with_steering = transformed[2*n_samples:]
    
    # Create overlay plot
    plt.figure(figsize=(12, 10))
    
    # Plot correct embeddings (green)
    plt.scatter(
        transformed_correct[:, 0],
        transformed_correct[:, 1],
        alpha=0.6,
        s=50,
        c='green',
        label='Correct Embeddings',
        marker='o',
        edgecolors='darkgreen',
        linewidths=0.5,
    )
    
    # Plot wrong embeddings without steering (blue)
    plt.scatter(
        transformed_wrong_no_steering[:, 0],
        transformed_wrong_no_steering[:, 1],
        alpha=0.5,
        s=40,
        c='blue',
        label='Wrong (Before Steering)',
        marker='s',
        edgecolors='darkblue',
        linewidths=0.5,
    )
    
    # Plot wrong embeddings with steering (red)
    plt.scatter(
        transformed_wrong_with_steering[:, 0],
        transformed_wrong_with_steering[:, 1],
        alpha=0.5,
        s=40,
        c='red',
        label='Wrong (After Steering)',
        marker='x',
        linewidths=1.5,
    )
    
    # Draw arrows from wrong no-steering to wrong with-steering for a few samples
    n_arrows = min(20, n_samples)  # Show arrows for up to 20 samples
    indices = np.linspace(0, n_samples - 1, n_arrows, dtype=int)
    for idx in indices:
        plt.arrow(
            transformed_wrong_no_steering[idx, 0],
            transformed_wrong_no_steering[idx, 1],
            transformed_wrong_with_steering[idx, 0] - transformed_wrong_no_steering[idx, 0],
            transformed_wrong_with_steering[idx, 1] - transformed_wrong_no_steering[idx, 1],
            head_width=0.05,
            head_length=0.05,
            fc='gray',
            ec='gray',
            alpha=0.4,
            length_includes_head=True,
            linestyle='--',
        )
    
    plt.title(f'Embedding Space Overlay: {method.upper()} (Layer {layer})', 
              fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    LOGGER.info("Saved overlay plot to %s", output_path)


def compute_shift_statistics(
    embeddings_correct: np.ndarray,
    embeddings_wrong_no_steering: np.ndarray,
    embeddings_wrong_with_steering: np.ndarray,
) -> Dict:
    """
    Compute statistics about how much embeddings shift with steering and distances to correct.
    
    Args:
        embeddings_correct: Correct embeddings [n_samples, hidden_dim]
        embeddings_wrong_no_steering: Wrong embeddings without steering [n_samples, hidden_dim]
        embeddings_wrong_with_steering: Wrong embeddings with steering [n_samples, hidden_dim]
    
    Returns:
        Dictionary with shift statistics
    """
    # Compute per-sample L2 distances from wrong (no steering) to wrong (with steering)
    shifts = embeddings_wrong_with_steering - embeddings_wrong_no_steering
    l2_distances_steering = np.linalg.norm(shifts, axis=1)
    
    # Compute distances from wrong (no steering) to correct
    wrong_to_correct_before = embeddings_correct - embeddings_wrong_no_steering
    l2_distances_to_correct_before = np.linalg.norm(wrong_to_correct_before, axis=1)
    
    # Compute distances from wrong (with steering) to correct
    wrong_to_correct_after = embeddings_correct - embeddings_wrong_with_steering
    l2_distances_to_correct_after = np.linalg.norm(wrong_to_correct_after, axis=1)
    
    # Compute cosine similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    cosine_sims_steering = np.array([
        cosine_similarity(embeddings_wrong_no_steering[i], embeddings_wrong_with_steering[i])
        for i in range(len(embeddings_wrong_no_steering))
    ])
    
    cosine_sims_to_correct_before = np.array([
        cosine_similarity(embeddings_wrong_no_steering[i], embeddings_correct[i])
        for i in range(len(embeddings_wrong_no_steering))
    ])
    
    cosine_sims_to_correct_after = np.array([
        cosine_similarity(embeddings_wrong_with_steering[i], embeddings_correct[i])
        for i in range(len(embeddings_wrong_with_steering))
    ])
    
    return {
        "steering_shift": {
            "mean_l2_shift": float(np.mean(l2_distances_steering)),
            "std_l2_shift": float(np.std(l2_distances_steering)),
            "median_l2_shift": float(np.median(l2_distances_steering)),
            "max_l2_shift": float(np.max(l2_distances_steering)),
            "mean_cosine_similarity": float(np.mean(cosine_sims_steering)),
            "std_cosine_similarity": float(np.std(cosine_sims_steering)),
            "min_cosine_similarity": float(np.min(cosine_sims_steering)),
        },
        "distance_to_correct_before_steering": {
            "mean_l2": float(np.mean(l2_distances_to_correct_before)),
            "std_l2": float(np.std(l2_distances_to_correct_before)),
            "median_l2": float(np.median(l2_distances_to_correct_before)),
            "mean_cosine_similarity": float(np.mean(cosine_sims_to_correct_before)),
            "std_cosine_similarity": float(np.std(cosine_sims_to_correct_before)),
        },
        "distance_to_correct_after_steering": {
            "mean_l2": float(np.mean(l2_distances_to_correct_after)),
            "std_l2": float(np.std(l2_distances_to_correct_after)),
            "median_l2": float(np.median(l2_distances_to_correct_after)),
            "mean_cosine_similarity": float(np.mean(cosine_sims_to_correct_after)),
            "std_cosine_similarity": float(np.std(cosine_sims_to_correct_after)),
        },
        "improvement": {
            "mean_l2_reduction": float(np.mean(l2_distances_to_correct_before - l2_distances_to_correct_after)),
            "mean_cosine_improvement": float(np.mean(cosine_sims_to_correct_after - cosine_sims_to_correct_before)),
        },
    }




def get_available_sample_ids(hidden_states_dir: Path, layer: int) -> List[str]:
    """Get list of sample IDs that have right, wrong, and steered embeddings for the given layer."""
    sample_ids = []
    if not hidden_states_dir.exists():
        return sample_ids
    
    for sample_dir in hidden_states_dir.iterdir():
        if sample_dir.is_dir():
            # Check if this sample has right, wrong, and steered embeddings for this layer
            right_path = sample_dir / f"right_layer{layer}.pt"
            wrong_path = sample_dir / f"wrong_layer{layer}.pt"
            steered_path = sample_dir / f"steered_layer{layer}.pt"
            if right_path.exists() and wrong_path.exists() and steered_path.exists():
                sample_ids.append(sample_dir.name)
    
    return sorted(sample_ids)


def process_layer(
    layer: int,
    args: argparse.Namespace,
    sample_ids: List[str],
) -> None:
    """Process a single layer: load embeddings and create plots."""
    LOGGER.info("=" * 80)
    LOGGER.info("Processing layer %d", layer)
    LOGGER.info("=" * 80)
    
    # Load embeddings from stored files
    LOGGER.info("Loading embeddings from stored files...")
    
    # Load right embeddings
    embeddings_correct, right_sample_ids = load_embeddings_from_files(
        args.hidden_states_dir,
        sample_ids,
        layer,
        "right",
        args.pooling_method,
    )
    
    # Load wrong embeddings
    embeddings_wrong_no_steering, wrong_sample_ids = load_embeddings_from_files(
        args.hidden_states_dir,
        sample_ids,
        layer,
        "wrong",
        args.pooling_method,
    )
    
    # Load steered embeddings
    embeddings_wrong_with_steering, steered_sample_ids = load_embeddings_from_files(
        args.hidden_states_dir,
        sample_ids,
        layer,
        "steered",
        args.pooling_method,
    )
    
    # Find common sample IDs across all three types
    common_sample_ids = list(set(right_sample_ids) & set(wrong_sample_ids) & set(steered_sample_ids))
    if not common_sample_ids:
        LOGGER.error("No common sample IDs found for layer %d (need right, wrong, and steered)", layer)
        return
    
    LOGGER.info("Using %d common samples for layer %d", len(common_sample_ids), layer)
    
    # Filter embeddings to common samples (maintain order)
    right_idx_map = {sid: i for i, sid in enumerate(right_sample_ids)}
    wrong_idx_map = {sid: i for i, sid in enumerate(wrong_sample_ids)}
    steered_idx_map = {sid: i for i, sid in enumerate(steered_sample_ids)}
    
    embeddings_correct = np.array([
        embeddings_correct[right_idx_map[sid]] for sid in common_sample_ids
    ])
    embeddings_wrong_no_steering = np.array([
        embeddings_wrong_no_steering[wrong_idx_map[sid]] for sid in common_sample_ids
    ])
    embeddings_wrong_with_steering = np.array([
        embeddings_wrong_with_steering[steered_idx_map[sid]] for sid in common_sample_ids
    ])
    
    # Verify all embeddings have same number of samples
    n_samples = len(embeddings_correct)
    if len(embeddings_wrong_no_steering) != n_samples or len(embeddings_wrong_with_steering) != n_samples:
        LOGGER.error(
            "Mismatch in number of samples: correct=%d, wrong_no_steering=%d, wrong_with_steering=%d",
            n_samples,
            len(embeddings_wrong_no_steering),
            len(embeddings_wrong_with_steering),
        )
        return
    
    LOGGER.info("Processing %d embeddings for layer %d", n_samples, layer)
    
    # Compute shift statistics
    stats = compute_shift_statistics(
        embeddings_correct,
        embeddings_wrong_no_steering,
        embeddings_wrong_with_steering
    )
    LOGGER.info("Shift statistics for layer %d:", layer)
    LOGGER.info("  Steering shift - Mean L2: %.4f ± %.4f", 
                stats["steering_shift"]["mean_l2_shift"], 
                stats["steering_shift"]["std_l2_shift"])
    LOGGER.info("  Distance to correct (before) - Mean L2: %.4f ± %.4f", 
                stats["distance_to_correct_before_steering"]["mean_l2"],
                stats["distance_to_correct_before_steering"]["std_l2"])
    LOGGER.info("  Distance to correct (after) - Mean L2: %.4f ± %.4f", 
                stats["distance_to_correct_after_steering"]["mean_l2"],
                stats["distance_to_correct_after_steering"]["std_l2"])
    LOGGER.info("  Improvement - Mean L2 reduction: %.4f", 
                stats["improvement"]["mean_l2_reduction"])
    LOGGER.info("  Improvement - Mean cosine improvement: %.4f", 
                stats["improvement"]["mean_cosine_improvement"])
    
    # Save statistics
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / f"layer{layer}_shift_statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    LOGGER.info("Saved statistics to %s", stats_path)
    
    # Create visualizations
    LOGGER.info("Creating visualizations for layer %d...", layer)
    
    # PCA overlay (main plot requested by user)
    pca_overlay_path = args.output_dir / f"layer{layer}_pca_overlay.png"
    plot_overlay_comparison(
        embeddings_correct,
        embeddings_wrong_no_steering,
        embeddings_wrong_with_steering,
        method="pca",
        output_path=pca_overlay_path,
        layer=layer,
    )
    
    # PCA comparison
    pca_comparison_path = args.output_dir / f"layer{layer}_pca_comparison.png"
    plot_embedding_comparison(
        embeddings_correct,
        embeddings_wrong_no_steering,
        embeddings_wrong_with_steering,
        method="pca",
        output_path=pca_comparison_path,
        layer=layer,
    )
    
    # UMAP overlay (if available)
    if umap is not None:
        umap_overlay_path = args.output_dir / f"layer{layer}_umap_overlay.png"
        plot_overlay_comparison(
            embeddings_correct,
            embeddings_wrong_no_steering,
            embeddings_wrong_with_steering,
            method="umap",
            output_path=umap_overlay_path,
            layer=layer,
        )
        
        # UMAP comparison
        umap_comparison_path = args.output_dir / f"layer{layer}_umap_comparison.png"
        plot_embedding_comparison(
            embeddings_correct,
            embeddings_wrong_no_steering,
            embeddings_wrong_with_steering,
            method="umap",
            output_path=umap_comparison_path,
            layer=layer,
        )
    else:
        LOGGER.warning("UMAP not available, skipping UMAP visualizations")
    
    LOGGER.info("Completed processing layer %d", layer)


def main() -> None:
    setup_logging()
    configure_hf_caches()
    args = parse_args()
    
    # Determine layers to process
    if args.layer is not None:
        # Backward compatibility: use --layer if provided
        layers_to_process = [args.layer]
    elif args.layers is not None:
        layers_to_process = sorted(set(args.layers))
    else:
        # Default layers
        layers_to_process = [26, 27, 28, 29, 30, 31]
    
    LOGGER.info("Processing layers: %s", layers_to_process)
    
    # Get sample IDs from available directories (use first layer to find samples)
    if layers_to_process:
        sample_ids = get_available_sample_ids(args.hidden_states_dir, layers_to_process[0])
        if not sample_ids:
            LOGGER.error("No sample IDs found in %s for layer %d with right, wrong, and steered embeddings", 
                        args.hidden_states_dir, layers_to_process[0])
            LOGGER.info("Note: Steered embeddings should be created using evaluate_steering.py with --save-steered-embeddings")
            return
        LOGGER.info("Found %d sample IDs with stored embeddings", len(sample_ids))
    else:
        LOGGER.error("No layers specified")
        return
    
    # Process each layer
    for layer in layers_to_process:
        try:
            process_layer(layer, args, sample_ids)
        except Exception as e:
            LOGGER.error("Error processing layer %d: %s", layer, e, exc_info=True)
            continue
    
    LOGGER.info("=" * 80)
    LOGGER.info("All layers processed. Results saved to %s", args.output_dir)
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()

