#!/usr/bin/env python3
"""Collect hidden activations for wrong/right chains."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from Levenshtein import opcodes
from tqdm import tqdm

from model_wrapper import ModelWrapper
from setup import (
    configure_hf_caches,
    setup_logging,
    build_teacher_forcing_text,
)


LOGGER = logging.getLogger("collect_hidden_states")


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
        "--output-dir",
        type=Path,
        default=Path("artifacts/hidden_states"),
        help="Directory for storing per-sample hidden state tensors.",
    )
    parser.add_argument(
        "--alignment-dir",
        type=Path,
        default=Path("artifacts/alignments"),
        help="Directory for saving token alignment metadata per sample.",
    )
    parser.add_argument(
        "--probe-data-dir",
        type=Path,
        default=Path("artifacts/probe_data"),
        help="Directory for saving probe training data (features and labels).",
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Identifier for a causal LM to evaluate (default mirrors Step 1).",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        required=True,
        type=int,
        help="List of 0-indexed transformer layer ids to capture (e.g., --layers 28 30 31).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of triples to process (for debugging).",
    )
    parser.add_argument(
        "--probe-max-samples",
        type=int,
        default=4000,
        help="Maximum number of matched token activations to use for probing and visualization per layer.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (defaults to cuda if available else cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility when subsampling tokens.",
    )
    parser.add_argument(
        "--pooling-method",
        type=str,
        default="mean",
        choices=["mean", "last_token", "per_token"],
        help="Pooling method: 'mean' (average all tokens), 'last_token' (final token), 'per_token' (each matched token).",
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


def align_tokens(
    model: ModelWrapper,
    wrong_input: Dict[str, torch.Tensor],
    right_input: Dict[str, torch.Tensor],
) -> List[Tuple[int, int]]:
    """
    Align tokens between wrong and right chains using Levenshtein edit distance.

    Args:
        model: ModelWrapper instance
        wrong_input: Tokenized wrong chain
        right_input: Tokenized right chain

    Returns:
        List of (wrong_idx, right_idx) tuples for matching tokens
    """
    wrong_tokens = model.token_strings(wrong_input["input_ids"][0])
    right_tokens = model.token_strings(right_input["input_ids"][0])

    alignment_opcodes = opcodes(wrong_tokens, right_tokens)
    matches: List[Tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in alignment_opcodes:
        if tag == "equal":
            matches.extend(zip(range(i1, i2), range(j1, j2)))
    return matches


def subsample_pairs(
    pairs: List[Tuple[int, int]], 
    max_samples: int, 
    rng: random.Random
) -> List[Tuple[int, int]]:
    """Subsample pairs if needed."""
    if len(pairs) <= max_samples:
        return pairs
    return rng.sample(pairs, max_samples)


def save_alignment(
    alignment_path: Path,
    wrong_tokens: List[str],
    right_tokens: List[str],
    matches: List[Tuple[int, int]]
) -> None:
    """Save token alignment to JSON."""
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "wrong_tokens": wrong_tokens,
        "right_tokens": right_tokens,
        "matches": [
            {"wrong_index": wrong_idx, "right_index": right_idx}
            for wrong_idx, right_idx in matches
        ],
    }
    alignment_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    setup_logging()
    configure_hf_caches()
    args = parse_args()
    rng = random.Random(args.seed)

    # Load triples
    triples = load_triples(args.triples_path)
    if not triples:
        LOGGER.error("No usable triples found (correct_chain missing?). Exiting.")
        return

    # Setup layers
    target_layers = sorted(set(args.layers))
    LOGGER.info("Target layers for hidden state capture: %s", target_layers)

    # Load model
    model = ModelWrapper(args.model_name, device=args.device)

    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.alignment_dir.mkdir(parents=True, exist_ok=True)
    args.probe_data_dir.mkdir(parents=True, exist_ok=True)

    # Probe data collection - track per layer
    probe_data_per_layer = {
        layer: {"features": [], "labels": [], "token_positions": [], "sample_ids": []} 
        for layer in target_layers
    }

    processed = 0
    skipped_missing_chain = 0

    for triple in tqdm(triples, desc="Capturing hidden states"):
        if args.max_samples is not None and processed >= args.max_samples:
            break

        if not triple.correct_chain.strip():
            skipped_missing_chain += 1
            continue

        # Build teacher forcing texts
        wrong_text = build_teacher_forcing_text(
            triple.prompt, triple.wrong_chain, triple.metadata
        )
        right_text = build_teacher_forcing_text(
            triple.prompt, triple.correct_chain, triple.metadata
        )

        # Tokenize
        wrong_inputs = model.tokenize(wrong_text, return_offsets_mapping=True)
        right_inputs = model.tokenize(right_text, return_offsets_mapping=True)

        # Get hidden states
        wrong_hidden = model.get_hidden_states(wrong_text, target_layers)
        right_hidden = model.get_hidden_states(right_text, target_layers)

        # Get tokens and align
        wrong_tokens = model.token_strings(wrong_inputs["input_ids"].squeeze(0))
        right_tokens = model.token_strings(right_inputs["input_ids"].squeeze(0))
        matches = align_tokens(model, wrong_inputs, right_inputs)

        # Save hidden states per sample
        sample_dir = args.output_dir / triple.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        for layer_id, tensor in wrong_hidden.items():
            torch.save(tensor, sample_dir / f"wrong_layer{layer_id}.pt")
        for layer_id, tensor in right_hidden.items():
            torch.save(tensor, sample_dir / f"right_layer{layer_id}.pt")

        # Save alignment
        alignment_path = args.alignment_dir / f"{triple.sample_id}.json"
        save_alignment(alignment_path, wrong_tokens, right_tokens, matches)

        # Collect probe data for all layers using specified pooling method
        for layer_id in target_layers:
            layer_data = probe_data_per_layer[layer_id]
            layer_tensor_wrong = wrong_hidden[layer_id]
            layer_tensor_right = right_hidden[layer_id]
            
            if args.pooling_method == "mean":
                # Mean pooling over entire sequence
                if len(layer_data["features"]) < args.probe_max_samples * 2:  # *2 for wrong+right
                    wrong_pooled = layer_tensor_wrong.mean(dim=0).numpy()
                    right_pooled = layer_tensor_right.mean(dim=0).numpy()
                    
                    layer_data["features"].append(wrong_pooled)
                    layer_data["labels"].append(0)  # 0 = wrong
                    layer_data["token_positions"].append(-1)  # -1 indicates pooled
                    layer_data["sample_ids"].append(triple.sample_id)
                    
                    layer_data["features"].append(right_pooled)
                    layer_data["labels"].append(1)  # 1 = right
                    layer_data["token_positions"].append(-1)
                    layer_data["sample_ids"].append(triple.sample_id)
            
            elif args.pooling_method == "last_token":
                # Last token (typically where the answer is generated)
                if len(layer_data["features"]) < args.probe_max_samples * 2:
                    wrong_last = layer_tensor_wrong[-1].numpy()
                    right_last = layer_tensor_right[-1].numpy()
                    
                    layer_data["features"].append(wrong_last)
                    layer_data["labels"].append(0)
                    layer_data["token_positions"].append(len(layer_tensor_wrong) - 1)
                    layer_data["sample_ids"].append(triple.sample_id)
                    
                    layer_data["features"].append(right_last)
                    layer_data["labels"].append(1)
                    layer_data["token_positions"].append(len(layer_tensor_right) - 1)
                    layer_data["sample_ids"].append(triple.sample_id)
            
            elif args.pooling_method == "per_token":
                # Per-token: use matched token positions
                if len(layer_data["features"]) < args.probe_max_samples:
                    # Subsample matched token positions
                    selected_matches = subsample_pairs(
                        matches, args.probe_max_samples - len(layer_data["features"]), rng
                    )
                    
                    for wrong_idx, right_idx in selected_matches:
                        # Add wrong token representation
                        layer_data["features"].append(layer_tensor_wrong[wrong_idx].numpy())
                        layer_data["labels"].append(0)  # 0 = wrong
                        layer_data["token_positions"].append(wrong_idx)
                        layer_data["sample_ids"].append(triple.sample_id)
                        
                        # Add right token representation
                        layer_data["features"].append(layer_tensor_right[right_idx].numpy())
                        layer_data["labels"].append(1)  # 1 = right
                        layer_data["token_positions"].append(right_idx)
                        layer_data["sample_ids"].append(triple.sample_id)

        processed += 1

    LOGGER.info("Processed %d triples (skipped %d missing correct chains)", 
               processed, skipped_missing_chain)

    # Save probe data for each layer
    for layer_id in target_layers:
        layer_data = probe_data_per_layer[layer_id]
        if layer_data["features"]:
            feature_matrix = np.stack(layer_data["features"])
            label_array = np.array(layer_data["labels"])
            position_array = np.array(layer_data["token_positions"])
            sample_id_array = np.array(layer_data["sample_ids"])
            
            probe_data_path = args.probe_data_dir / f"layer{layer_id}_probe_data.npz"
            np.savez(
                probe_data_path,
                features=feature_matrix,
                labels=label_array,
                token_positions=position_array,
                sample_ids=sample_id_array,
                layer=layer_id,
            )
            LOGGER.info("Saved probe data for layer %d to %s (%d samples)", 
                       layer_id, probe_data_path, len(layer_data["features"]))
        else:
            LOGGER.warning("No probe samples collected for layer %d.", layer_id)


if __name__ == "__main__":
    main()
