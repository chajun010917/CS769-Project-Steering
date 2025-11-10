#!/usr/bin/env python3
"""Compare hidden state embeddings with and without steering to check if they're in the same space."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

try:
    import umap
except ImportError:
    umap = None

from model_wrapper import ModelWrapper
from setup import (
    configure_hf_caches,
    setup_logging,
    reconstruct_prompt,
)

LOGGER = logging.getLogger("compare_steering_embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--triples-path",
        type=Path,
        required=True,
        help="Path to JSONL file produced by prepare_triples.py",
    )
    parser.add_argument(
        "--steering-vectors-dir",
        type=Path,
        default=Path("artifacts/steering_vectors"),
        help="Directory containing steering vectors (layer{layer_id}_steering_vector.npy files).",
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Identifier for a causal LM.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer ID to analyze (e.g., 26).",
    )
    parser.add_argument(
        "--steering-coefficient",
        type=float,
        default=1.0,
        help="Multiplier for steering vectors (default: 1.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/steering_embedding_comparison"),
        help="Directory for saving comparison plots.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of triples to process (for debugging).",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'.",
        help="System message used when the tokenizer supports chat templates.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (defaults to cuda if available else cpu).",
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


def load_triples(path: Path) -> List[Dict]:
    """Load triples from JSONL file."""
    triples: List[Dict] = []
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            payload = json.loads(line)
            if payload.get("correct_chain"):
                triples.append(payload)
    return triples


def load_steering_vector(steering_dir: Path, layer_id: int) -> Optional[np.ndarray]:
    """Load steering vector for specified layer."""
    steering_path = steering_dir / f"layer{layer_id}_steering_vector.npy"
    if not steering_path.exists():
        LOGGER.warning("Steering vector not found for layer %d at %s", layer_id, steering_path)
        return None
    steering_vector = np.load(steering_path)
    LOGGER.info(
        "Loaded steering vector for layer %d (shape: %s, norm: %.4f)",
        layer_id,
        steering_vector.shape,
        np.linalg.norm(steering_vector),
    )
    return steering_vector


def collect_hidden_states(
    model: ModelWrapper,
    triples: List[Dict],
    layer: int,
    steering_vector: Optional[np.ndarray] = None,
    steering_coefficient: float = 1.0,
    system_prompt: str = "",
    pooling_method: str = "last_token",
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Collect hidden states from the prompt's last token.
    For steering case, we add the steering vector to simulate the effect.
    
    Note: This captures hidden states from the prompt, not from generated tokens.
    To see the actual effect during generation, steering is applied additively to
    simulate what happens when steering is injected during generation.
    
    Args:
        model: Model wrapper
        triples: List of triples to process
        layer: Layer ID to collect from
        steering_vector: Optional steering vector to apply (added to hidden states)
        steering_coefficient: Multiplier for steering vector
        system_prompt: System prompt for chat template (used for formatting)
        pooling_method: How to pool hidden states (last_token or mean)
        max_samples: Optional limit on number of samples
    
    Returns:
        Array of hidden states [n_samples, hidden_dim]
    """
    if max_samples:
        triples = triples[:max_samples]
    
    hidden_states_list = []
    
    for triple in tqdm(triples, desc="Collecting hidden states"):
        prompt_text = triple.get("prompt", "")
        metadata = triple.get("metadata", {})
        
        # Reconstruct prompt
        formatted_prompt = reconstruct_prompt(metadata, prompt_text)
        
        # Apply chat template if available and system prompt provided (for consistency with generation)
        if hasattr(model.tokenizer, "apply_chat_template") and system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_prompt},
            ]
            formatted_prompt = model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        # Get hidden states from prompt
        hidden_states = model.get_hidden_states(formatted_prompt, target_layers=[layer])
        hidden_state = hidden_states[layer]  # [seq_len, hidden_dim]
        
        # Apply pooling
        if pooling_method == "last_token":
            pooled = hidden_state[-1].cpu().numpy()  # [hidden_dim]
        elif pooling_method == "mean":
            pooled = hidden_state.mean(dim=0).cpu().numpy()  # [hidden_dim]
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")
        
        # Apply steering vector if provided (simulates steering effect)
        if steering_vector is not None:
            pooled = pooled + (steering_vector * steering_coefficient)
        
        hidden_states_list.append(pooled)
    
    return np.array(hidden_states_list)


def plot_embedding_comparison(
    embeddings_no_steering: np.ndarray,
    embeddings_with_steering: np.ndarray,
    method: str,
    output_path: Path,
    layer: int,
    steering_coefficient: float,
) -> None:
    """
    Plot comparison of embeddings with and without steering.
    
    Args:
        embeddings_no_steering: Embeddings without steering [n_samples, hidden_dim]
        embeddings_with_steering: Embeddings with steering [n_samples, hidden_dim]
        method: Visualization method ('pca' or 'umap')
        output_path: Path to save plot
        layer: Layer ID
        steering_coefficient: Steering coefficient used
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine embeddings for fitting
    combined_embeddings = np.vstack([embeddings_no_steering, embeddings_with_steering])
    
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
    
    # Split back into no-steering and with-steering
    n_samples = len(embeddings_no_steering)
    transformed_no_steering = transformed[:n_samples]
    transformed_with_steering = transformed[n_samples:]
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Without steering
    ax1 = axes[0]
    ax1.scatter(
        transformed_no_steering[:, 0],
        transformed_no_steering[:, 1],
        alpha=0.6,
        s=30,
        c='blue',
        label='Without Steering',
    )
    ax1.set_title(f'Embeddings Without Steering (Layer {layer})', fontsize=14, fontweight='bold')
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: With steering
    ax2 = axes[1]
    ax2.scatter(
        transformed_with_steering[:, 0],
        transformed_with_steering[:, 1],
        alpha=0.6,
        s=30,
        c='red',
        label=f'With Steering (coeff={steering_coefficient})',
    )
    ax2.set_title(f'Embeddings With Steering (Layer {layer})', fontsize=14, fontweight='bold')
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Embedding Space Comparison: {method.upper()} (Layer {layer})', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    LOGGER.info("Saved comparison plot to %s", output_path)


def plot_overlay_comparison(
    embeddings_no_steering: np.ndarray,
    embeddings_with_steering: np.ndarray,
    method: str,
    output_path: Path,
    layer: int,
    steering_coefficient: float,
) -> None:
    """
    Plot overlay comparison showing both embeddings in the same space.
    
    Args:
        embeddings_no_steering: Embeddings without steering [n_samples, hidden_dim]
        embeddings_with_steering: Embeddings with steering [n_samples, hidden_dim]
        method: Visualization method ('pca' or 'umap')
        output_path: Path to save plot
        layer: Layer ID
        steering_coefficient: Steering coefficient used
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine embeddings for fitting
    combined_embeddings = np.vstack([embeddings_no_steering, embeddings_with_steering])
    
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
    
    # Split back into no-steering and with-steering
    n_samples = len(embeddings_no_steering)
    transformed_no_steering = transformed[:n_samples]
    transformed_with_steering = transformed[n_samples:]
    
    # Create overlay plot
    plt.figure(figsize=(10, 8))
    
    plt.scatter(
        transformed_no_steering[:, 0],
        transformed_no_steering[:, 1],
        alpha=0.5,
        s=40,
        c='blue',
        label='Without Steering',
        marker='o',
    )
    plt.scatter(
        transformed_with_steering[:, 0],
        transformed_with_steering[:, 1],
        alpha=0.5,
        s=40,
        c='red',
        label=f'With Steering (coeff={steering_coefficient})',
        marker='x',
    )
    
    # Draw arrows from no-steering to with-steering for a few samples
    n_arrows = min(20, n_samples)  # Show arrows for up to 20 samples
    indices = np.linspace(0, n_samples - 1, n_arrows, dtype=int)
    for idx in indices:
        plt.arrow(
            transformed_no_steering[idx, 0],
            transformed_no_steering[idx, 1],
            transformed_with_steering[idx, 0] - transformed_no_steering[idx, 0],
            transformed_with_steering[idx, 1] - transformed_no_steering[idx, 1],
            head_width=0.05,
            head_length=0.05,
            fc='gray',
            ec='gray',
            alpha=0.3,
            length_includes_head=True,
        )
    
    plt.title(f'Embedding Space Overlay: {method.upper()} (Layer {layer})', 
              fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    LOGGER.info("Saved overlay plot to %s", output_path)


def compute_shift_statistics(
    embeddings_no_steering: np.ndarray,
    embeddings_with_steering: np.ndarray,
) -> Dict:
    """
    Compute statistics about how much embeddings shift with steering.
    
    Args:
        embeddings_no_steering: Embeddings without steering [n_samples, hidden_dim]
        embeddings_with_steering: Embeddings with steering [n_samples, hidden_dim]
    
    Returns:
        Dictionary with shift statistics
    """
    # Compute per-sample L2 distances
    shifts = embeddings_with_steering - embeddings_no_steering
    l2_distances = np.linalg.norm(shifts, axis=1)
    
    # Compute cosine similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    cosine_sims = np.array([
        cosine_similarity(embeddings_no_steering[i], embeddings_with_steering[i])
        for i in range(len(embeddings_no_steering))
    ])
    
    return {
        "mean_l2_shift": float(np.mean(l2_distances)),
        "std_l2_shift": float(np.std(l2_distances)),
        "median_l2_shift": float(np.median(l2_distances)),
        "max_l2_shift": float(np.max(l2_distances)),
        "mean_cosine_similarity": float(np.mean(cosine_sims)),
        "std_cosine_similarity": float(np.std(cosine_sims)),
        "min_cosine_similarity": float(np.min(cosine_sims)),
    }


def main() -> None:
    setup_logging()
    configure_hf_caches()
    args = parse_args()
    
    # Load triples
    triples = load_triples(args.triples_path)
    if not triples:
        LOGGER.error("No usable triples found. Exiting.")
        return
    
    if args.max_samples:
        triples = triples[:args.max_samples]
    
    LOGGER.info("Loaded %d triples for comparison", len(triples))
    
    # Load steering vector
    steering_vector = load_steering_vector(args.steering_vectors_dir, args.layer)
    if steering_vector is None:
        LOGGER.error("Steering vector not found. Exiting.")
        return
    
    # Load model
    model = ModelWrapper(args.model_name, device=args.device)
    
    # Collect hidden states without steering
    LOGGER.info("Collecting hidden states without steering...")
    embeddings_no_steering = collect_hidden_states(
        model=model,
        triples=triples,
        layer=args.layer,
        steering_vector=None,
        system_prompt=args.system_prompt,
        pooling_method=args.pooling_method,
        max_samples=args.max_samples,
    )
    LOGGER.info("Collected %d embeddings without steering (shape: %s)", 
                len(embeddings_no_steering), embeddings_no_steering.shape)
    
    # Collect hidden states with steering (steering vector added to simulate effect)
    LOGGER.info("Collecting hidden states with steering (simulated by adding steering vector)...")
    embeddings_with_steering = collect_hidden_states(
        model=model,
        triples=triples,
        layer=args.layer,
        steering_vector=steering_vector,
        steering_coefficient=args.steering_coefficient,
        system_prompt=args.system_prompt,
        pooling_method=args.pooling_method,
        max_samples=args.max_samples,
    )
    LOGGER.info("Collected %d embeddings with steering (shape: %s)", 
                len(embeddings_with_steering), embeddings_with_steering.shape)
    
    # Compute shift statistics
    stats = compute_shift_statistics(embeddings_no_steering, embeddings_with_steering)
    LOGGER.info("Shift statistics:")
    LOGGER.info("  Mean L2 shift: %.4f ± %.4f", stats["mean_l2_shift"], stats["std_l2_shift"])
    LOGGER.info("  Median L2 shift: %.4f", stats["median_l2_shift"])
    LOGGER.info("  Max L2 shift: %.4f", stats["max_l2_shift"])
    LOGGER.info("  Mean cosine similarity: %.4f ± %.4f", 
                stats["mean_cosine_similarity"], stats["std_cosine_similarity"])
    LOGGER.info("  Min cosine similarity: %.4f", stats["min_cosine_similarity"])
    
    # Save statistics
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / f"layer{args.layer}_shift_statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    LOGGER.info("Saved statistics to %s", stats_path)
    
    # Create visualizations
    LOGGER.info("Creating visualizations...")
    
    # PCA comparison
    pca_comparison_path = args.output_dir / f"layer{args.layer}_pca_comparison.png"
    plot_embedding_comparison(
        embeddings_no_steering,
        embeddings_with_steering,
        method="pca",
        output_path=pca_comparison_path,
        layer=args.layer,
        steering_coefficient=args.steering_coefficient,
    )
    
    # PCA overlay
    pca_overlay_path = args.output_dir / f"layer{args.layer}_pca_overlay.png"
    plot_overlay_comparison(
        embeddings_no_steering,
        embeddings_with_steering,
        method="pca",
        output_path=pca_overlay_path,
        layer=args.layer,
        steering_coefficient=args.steering_coefficient,
    )
    
    # UMAP comparison (if available)
    if umap is not None:
        umap_comparison_path = args.output_dir / f"layer{args.layer}_umap_comparison.png"
        plot_embedding_comparison(
            embeddings_no_steering,
            embeddings_with_steering,
            method="umap",
            output_path=umap_comparison_path,
            layer=args.layer,
            steering_coefficient=args.steering_coefficient,
        )
        
        # UMAP overlay
        umap_overlay_path = args.output_dir / f"layer{args.layer}_umap_overlay.png"
        plot_overlay_comparison(
            embeddings_no_steering,
            embeddings_with_steering,
            method="umap",
            output_path=umap_overlay_path,
            layer=args.layer,
            steering_coefficient=args.steering_coefficient,
        )
    else:
        LOGGER.warning("UMAP not available, skipping UMAP visualizations")
    
    LOGGER.info("All visualizations saved to %s", args.output_dir)


if __name__ == "__main__":
    main()

