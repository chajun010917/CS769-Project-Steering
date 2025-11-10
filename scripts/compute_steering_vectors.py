#!/usr/bin/env python3
"""Compute steering vectors from last_token hidden states (right - wrong, averaged)."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from model_wrapper import ModelWrapper
from setup import (
    configure_hf_caches,
    setup_logging,
    build_teacher_forcing_text,
    reconstruct_prompt,
)


LOGGER = logging.getLogger("compute_steering_vectors")


@dataclass
class Triple:
    sample_id: str
    prompt: str
    wrong_chain: str
    correct_chain: str
    wrong_answer: str
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
        "--token-selection-method",
        choices=["last_token", "gradient"],
        default="last_token",
        help="How to choose the token position for computing steering vectors.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (defaults to cuda if available else cpu).",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'.",
        help="System message used when the tokenizer supports chat templates.",
    )
    return parser.parse_args()


def load_triples(path: Path) -> List[Triple]:
    """Load triples from JSONL or JSON file."""
    triples: List[Triple] = []
    with path.open("r", encoding="utf-8") as src:
        first_chunk = src.read(1)

    if not first_chunk:
        LOGGER.error("Triples file %s is empty.", path)
        return triples

    is_json_array = first_chunk in ("[", "{")

    if is_json_array:
        with path.open("r", encoding="utf-8") as src:
            try:
                payloads = json.load(src)
            except json.JSONDecodeError as exc:
                LOGGER.error("Failed to parse JSON file %s: %s", path, exc)
                return triples

        if isinstance(payloads, dict):
            payloads = payloads.get("records") or payloads.get("data") or []
            if not isinstance(payloads, list):
                LOGGER.error("JSON file %s does not contain a list of records.", path)
                return triples
    else:
        payloads = []
        with path.open("r", encoding="utf-8") as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    payloads.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    LOGGER.warning("Skipping malformed JSON line: %s (error: %s)", line[:200], exc)

    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        correct_chain = payload.get("correct_chain")
        if not correct_chain:
            continue
        triples.append(
            Triple(
                sample_id=str(payload.get("sample_id")),
                prompt=str(payload.get("prompt", "")),
                wrong_chain=str(payload.get("wrong_chain", "")),
                correct_chain=str(correct_chain),
                wrong_answer=str(payload.get("wrong_answer", "")),
                correct_answer=str(payload.get("correct_answer", "")),
                metadata=dict(payload.get("metadata", {})),
            )
        )
    return triples


def encode_answer_token_id(model: ModelWrapper, answer: str) -> Optional[int]:
    """Encode an answer string and return the final token id (if available)."""
    if not answer:
        return None
    cleaned = answer.strip()
    if not cleaned:
        return None
    token_ids = model.tokenizer.encode(cleaned, add_special_tokens=False)
    if token_ids:
        return int(token_ids[-1])
    # Try using only the first character (useful for single letter answers)
    if len(cleaned) > 1:
        token_ids = model.tokenizer.encode(cleaned[0], add_special_tokens=False)
        if token_ids:
            return int(token_ids[-1])
    return None


def compute_prompt_token_length(model: ModelWrapper, prompt_text: str) -> int:
    """Return the number of tokens corresponding to the prompt (used to mask gradients)."""
    encoded = model.tokenize(prompt_text)
    return int(encoded["input_ids"].shape[1])


def select_tokens_via_gradient(
    model: ModelWrapper,
    forward_outputs: Dict[str, Any],
    target_layers: List[int],
    prompt_token_count: int,
    correct_token_id: Optional[int],
    wrong_token_id: Optional[int],
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    Use gradient of the correct answer logit (optionally contrasted with wrong answer logit)
    to identify the most influential token per layer.
    """
    if correct_token_id is None:
        raise ValueError("Correct answer token id unavailable for gradient-based selection.")

    logits: torch.Tensor = forward_outputs["logits"]
    last_position = logits.shape[0] - 1

    score = logits[last_position, correct_token_id]
    if wrong_token_id is not None and wrong_token_id != correct_token_id:
        score = score - logits[last_position, wrong_token_id]

    model.model.zero_grad(set_to_none=True)
    score.backward(retain_graph=False)

    selected_positions: Dict[int, int] = {}
    selected_norms: Dict[int, float] = {}

    for layer_id in target_layers:
        layer_tensor: torch.Tensor = forward_outputs["hidden_states"][layer_id]
        grad_tensor = layer_tensor.grad
        if grad_tensor is None:
            raise ValueError(f"Gradient not available for layer {layer_id}.")

        grad_tensor = grad_tensor.squeeze(0)
        grad_norm = grad_tensor.norm(p=2, dim=1)
        if grad_norm.numel() == 0:
            raise ValueError(f"No tokens available for layer {layer_id}.")

        mask_upto = min(prompt_token_count, grad_norm.shape[0] - 1)
        if mask_upto > 0:
            grad_norm[:mask_upto] = -float("inf")

        token_idx_tensor = torch.argmax(grad_norm)
        token_idx = int(token_idx_tensor.item())

        if token_idx < 0 or token_idx >= grad_norm.shape[0]:
            raise ValueError(f"Invalid token index {token_idx} for layer {layer_id}.")

        selected_positions[layer_id] = token_idx
        selected_norms[layer_id] = float(grad_norm[token_idx_tensor].item())

    model.model.zero_grad(set_to_none=True)

    return selected_positions, selected_norms


def summarize_numeric(values: List[float]) -> Dict[str, Any]:
    """Compute basic summary statistics for a list of numeric values."""
    if not values:
        return {"count": 0}
    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


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
    token_positions_per_layer: Dict[int, List[int]] = {
        layer: [] for layer in target_layers
    }
    gradient_norms_per_layer: Dict[int, List[float]] = {
        layer: [] for layer in target_layers
    }

    processed = 0
    skipped = 0
    gradient_fallbacks = 0

    use_gradient = args.token_selection_method == "gradient"

    for triple in tqdm(triples, desc="Computing steering vectors"):
        if not triple.correct_chain.strip():
            skipped += 1
            continue

        # Build texts
        wrong_text = build_teacher_forcing_text(
            triple.prompt, triple.wrong_chain, triple.metadata
        )
        right_text = build_teacher_forcing_text(
            triple.prompt, triple.correct_chain, triple.metadata
        )

        formatted_prompt = reconstruct_prompt(triple.metadata, triple.prompt)
        prompt_token_count = compute_prompt_token_length(model, formatted_prompt)

        # Forward passes
        wrong_forward = model.forward_with_hidden_states(
            wrong_text,
            target_layers=target_layers,
            enable_grad=use_gradient,
            system_prompt=args.system_prompt,
        )
        right_forward = model.forward_with_hidden_states(
            right_text,
            target_layers=target_layers,
            enable_grad=False,
            system_prompt=args.system_prompt,
        )

        selected_positions: Dict[int, int] = {}
        gradient_norms: Dict[int, float] = {}
        gradient_used = False

        if use_gradient:
            try:
                correct_token_id = encode_answer_token_id(model, triple.correct_answer)
                wrong_token_id = encode_answer_token_id(model, triple.wrong_answer)
                selected_positions, gradient_norms = select_tokens_via_gradient(
                    model=model,
                    forward_outputs=wrong_forward,
                    target_layers=target_layers,
                    prompt_token_count=prompt_token_count,
                    correct_token_id=correct_token_id,
                    wrong_token_id=wrong_token_id,
                )
                gradient_used = True
            except Exception as exc:
                gradient_fallbacks += 1
                LOGGER.warning(
                    "Gradient-based token selection failed for sample %s: %s. Falling back to last token.",
                    triple.sample_id,
                    exc,
                )
                selected_positions = {}
                gradient_used = False
                model.model.zero_grad(set_to_none=True)

        # Fallback or last token selection
        for layer_id in target_layers:
            layer_states = wrong_forward["hidden_states"][layer_id]
            seq_len = layer_states.shape[0]
            default_idx = max(0, seq_len - 1)
            selected_positions.setdefault(layer_id, default_idx)

        # Compute differences using selected tokens
        for layer_id in target_layers:
            wrong_states = wrong_forward["hidden_states"][layer_id].squeeze(0)
            right_states = right_forward["hidden_states"][layer_id].squeeze(0)

            token_idx = selected_positions[layer_id]
            token_idx = max(0, min(token_idx, wrong_states.shape[0] - 1))
            right_idx = max(0, min(token_idx, right_states.shape[0] - 1))

            wrong_vec = wrong_states[token_idx].detach().float().cpu().numpy()
            right_vec = right_states[right_idx].detach().float().cpu().numpy()

            diff = right_vec - wrong_vec
            difference_vectors_per_layer[layer_id].append(diff)
            token_positions_per_layer[layer_id].append(token_idx)

            if gradient_used and layer_id in gradient_norms:
                gradient_norms_per_layer[layer_id].append(gradient_norms[layer_id])

        processed += 1

    LOGGER.info(
        "Processed %d triples (skipped %d, gradient fallbacks %d)",
        processed,
        skipped,
        gradient_fallbacks,
    )

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
        "token_selection_method": args.token_selection_method,
        "token_position_summary": {
            layer_id: summarize_numeric(token_positions_per_layer[layer_id])
            for layer_id in target_layers
        },
        "gradient_fallbacks": gradient_fallbacks,
    }
    if args.token_selection_method == "gradient":
        metadata["gradient_norm_summary"] = {
            layer_id: summarize_numeric(gradient_norms_per_layer[layer_id])
            for layer_id in target_layers
        }
    metadata_path = args.output_dir / "steering_vectors_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved metadata to %s", metadata_path)

    LOGGER.info("Steering vector computation complete!")


if __name__ == "__main__":
    main()

