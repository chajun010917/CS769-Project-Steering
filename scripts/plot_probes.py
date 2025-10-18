#!/usr/bin/env python3
"""Plot probe analysis visualizations: PCA, UMAP, CAA projections, and cluster overlays."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from setup import setup_logging


LOGGER = logging.getLogger("plot_probes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Directory containing computed probe analysis data (output from compute_probes.py).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer number to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/hidden_state_viz"),
        help="Directory for saving plots.",
    )
    return parser.parse_args()



def plot_cluster_overlay(
    points: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
) -> None:
    """
    Save a 2D scatter plot with different markers for each class.

    Args:
        points: 2D points (n_samples, 2)
        labels: Binary labels
        output_path: Path to save plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    label_palette = {0: "red", 1: "blue"}

    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            c=label_palette.get(label, "gray"),
            marker="o" if label == 0 else "x",
            s=30,
            alpha=0.7,
            label="wrong" if label == 0 else "right",
        )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    LOGGER.info("Saved cluster overlay to %s", output_path)


def main() -> None:
    setup_logging()
    args = parse_args()

    layer = args.layer

    # Check if analysis directory exists
    if not args.analysis_dir.exists():
        LOGGER.error("Analysis directory not found: %s", args.analysis_dir)
        return

    # Load PCA data
    pca_data_path = args.analysis_dir / f"layer{layer}_pca_data.npz"
    if pca_data_path.exists():
        LOGGER.info("Loading PCA data from %s", pca_data_path)
        pca_data = np.load(pca_data_path)
        pca_transformed = pca_data["transformed"]
        pca_labels = pca_data["labels"]
        
        # Plot PCA cluster overlay
        overlay_dir = args.output_dir / "cluster_overlays"
        pca_cluster_path = overlay_dir / f"layer{layer}_pca_clusters.png"
        plot_cluster_overlay(pca_transformed, pca_labels, pca_cluster_path, "PCA Separation")
    else:
        LOGGER.warning("PCA data not found: %s", pca_data_path)

    # Load UMAP data
    umap_data_path = args.analysis_dir / f"layer{layer}_umap_data.npz"
    if umap_data_path.exists():
        LOGGER.info("Loading UMAP data from %s", umap_data_path)
        umap_data = np.load(umap_data_path)
        umap_transformed = umap_data["transformed"]
        umap_labels = umap_data["labels"]
        
        # Plot UMAP cluster overlay
        overlay_dir = args.output_dir / "cluster_overlays"
        umap_cluster_path = overlay_dir / f"layer{layer}_umap_clusters.png"
        plot_cluster_overlay(umap_transformed, umap_labels, umap_cluster_path, "UMAP Separation")
    else:
        LOGGER.warning("UMAP data not found: %s", umap_data_path)

    LOGGER.info("All plots saved to %s", args.output_dir)


if __name__ == "__main__":
    main()


'''
def plot_caa_histogram(
    projections: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot histogram of feature projections onto CAA direction.
    
    Args:
        projections: 1D array of projections onto CAA vector
        labels: Binary labels (0=wrong, 1=right)
        output_path: Path to save plot
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    wrong_proj = projections[labels == 0]
    right_proj = projections[labels == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(wrong_proj, bins=50, alpha=0.6, label='Wrong', color='red', density=True)
    plt.hist(right_proj, bins=50, alpha=0.6, label='Right', color='blue', density=True)
    
    plt.xlabel('Projection onto CAA Direction', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Feature Projections onto CAA Vector', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    LOGGER.info("Saved CAA projection histogram to %s", output_path)


def plot_caa_box_comparison(
    projections: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot box plot comparing distributions along CAA direction.
    
    Args:
        projections: 1D array of projections onto CAA vector
        labels: Binary labels
        output_path: Path to save plot
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    wrong_proj = projections[labels == 0]
    right_proj = projections[labels == 1]
    
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        [wrong_proj, right_proj],
        labels=['Wrong', 'Right'],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', alpha=0.7),
        medianprops=dict(color='red', linewidth=2),
    )
    
    plt.ylabel('Projection onto CAA Direction', fontsize=12)
    plt.title('Distribution Comparison Along CAA Vector', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    LOGGER.info("Saved CAA box plot to %s", output_path)


def plot_caa_density(
    projections: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot density curves for correct/incorrect along CAA direction.
    
    Args:
        projections: 1D array of projections
        labels: Binary labels
        output_path: Path to save plot
    """
    try:
        from scipy import stats
    except ImportError:
        LOGGER.warning("scipy not available, skipping density plot")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    wrong_proj = projections[labels == 0]
    right_proj = projections[labels == 1]
    
    # Compute KDEs
    x_range = np.linspace(projections.min(), projections.max(), 200)
    kde_wrong = stats.gaussian_kde(wrong_proj)
    kde_right = stats.gaussian_kde(right_proj)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, kde_wrong(x_range), 'r-', linewidth=2, label='Wrong', alpha=0.8)
    plt.plot(x_range, kde_right(x_range), 'b-', linewidth=2, label='Right', alpha=0.8)
    plt.fill_between(x_range, kde_wrong(x_range), alpha=0.3, color='red')
    plt.fill_between(x_range, kde_right(x_range), alpha=0.3, color='blue')
    
    plt.xlabel('Projection onto CAA Direction', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Kernel Density Estimate Along CAA Vector', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    LOGGER.info("Saved CAA density plot to %s", output_path)
    '''