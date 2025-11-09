#!/usr/bin/env python3
"""Compute steering vectors from last_token hidden states (right - wrong, averaged)."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from model_wrapper import ModelWrapper
from setup import configure_hf_caches, setup_logging, build_teacher_forcing_text


LOGGER = logging.getLogger("compute_steering_vectors")


@dataclass
class Triple:
    sample_id: str
    prompt: str
    wrong_chain: str
    correct_chain: str
    correct_answer: str
    metadata: Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--triples-path",
        type=Path,
        required=True,
        help="Path to JSONL file produced by prepare_triples.py",
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Identifier for a causal LM (default mirrors Step 1).",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        required=True,
        type=int,
        help="List of 0-indexed transformer layer ids to compute steering vectors for (e.g., --layers 28 30 31).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/steering_vectors"),
        help="Directory for saving steering vectors.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of triples to process (for debugging).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (defaults to cuda if available else cpu).",
    )
    return parser.parse_args()


def load_triples(path: Path) -> List[Triple]:
    """Load triples from JSONL file."""
    triples: List[Triple] = []
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            payload = json.loads(line)
            correct_chain = payload.get("correct_chain")
            if not correct_chain:
                continue
            triples.append(
                Triple(
                    sample_id=str(payload.get("sample_id")),
                    prompt=str(payload.get("prompt", "")),
                    wrong_chain=str(payload.get("wrong_chain", "")),
                    correct_chain=str(correct_chain),
                    correct_answer=str(payload.get("correct_answer", "")),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
    return triples


def main() -> None:
    setup_logging()
    configure_hf_caches()
    args = parse_args()

    # Load triples
    triples = load_triples(args.triples_path)
    if not triples:
        LOGGER.error("No usable triples found (correct_chain missing?). Exiting.")
        return

    if args.max_samples:
        triples = triples[:args.max_samples]

    LOGGER.info("Loaded %d triples", len(triples))

    # Setup layers
    target_layers = sorted(set(args.layers))
    LOGGER.info("Target layers for steering vector computation: %s", target_layers)

    # Load model
    model = ModelWrapper(args.model_name, device=args.device)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect difference vectors per layer
    # For each layer, we'll collect: right_last_token - wrong_last_token
    difference_vectors_per_layer: Dict[int, List[np.ndarray]] = {
        layer: [] for layer in target_layers
    }

    processed = 0
    skipped = 0

    for triple in tqdm(triples, desc="Computing steering vectors"):
        if not triple.correct_chain.strip():
            skipped += 1
            continue

        # Build teacher forcing texts
        wrong_text = build_teacher_forcing_text(
            triple.prompt, triple.wrong_chain, triple.metadata
        )
        right_text = build_teacher_forcing_text(
            triple.prompt, triple.correct_chain, triple.metadata
        )

        # Get hidden states at last token for all target layers
        wrong_hidden = model.get_hidden_states(wrong_text, target_layers)
        right_hidden = model.get_hidden_states(right_text, target_layers)

        # Extract last token representations and compute differences
        for layer_id in target_layers:
            wrong_last = wrong_hidden[layer_id][-1].numpy()  # Last token: [hidden_dim]
            right_last = right_hidden[layer_id][-1].numpy()  # Last token: [hidden_dim]
            
            # Compute difference: right - wrong (steering from wrong to right)
            diff = right_last - wrong_last
            difference_vectors_per_layer[layer_id].append(diff)

        processed += 1

    LOGGER.info("Processed %d triples (skipped %d)", processed, skipped)

    # Compute steering vectors by averaging differences
    steering_vectors: Dict[int, np.ndarray] = {}
    for layer_id in target_layers:
        if not difference_vectors_per_layer[layer_id]:
            LOGGER.warning("No difference vectors collected for layer %d", layer_id)
            continue

        # Stack all difference vectors and compute mean
        differences = np.stack(difference_vectors_per_layer[layer_id])  # [n_samples, hidden_dim]
        steering_vector = differences.mean(axis=0)  # [hidden_dim]
        
        steering_vectors[layer_id] = steering_vector
        
        LOGGER.info(
            "Layer %d: Computed steering vector from %d samples (shape: %s, norm: %.4f)",
            layer_id,
            len(difference_vectors_per_layer[layer_id]),
            steering_vector.shape,
            np.linalg.norm(steering_vector),
        )

        # Save steering vector
        steering_path = args.output_dir / f"layer{layer_id}_steering_vector.npy"
        np.save(steering_path, steering_vector)
        LOGGER.info("Saved steering vector for layer %d to %s", layer_id, steering_path)

    # Save metadata
    metadata = {
        "model_name": args.model_name,
        "layers": target_layers,
        "num_triples": processed,
        "num_samples_per_layer": {
            layer_id: len(difference_vectors_per_layer[layer_id])
            for layer_id in target_layers
        },
        "steering_vector_norms": {
            layer_id: float(np.linalg.norm(steering_vectors[layer_id]))
            for layer_id in steering_vectors.keys()
        },
    }
    metadata_path = args.output_dir / "steering_vectors_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved metadata to %s", metadata_path)

    LOGGER.info("Steering vector computation complete!")


if __name__ == "__main__":
    main()

