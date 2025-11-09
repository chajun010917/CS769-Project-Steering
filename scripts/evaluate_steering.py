#!/usr/bin/env python3
"""Evaluate steering vectors on triples to see if they improve predictions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from model_wrapper import ModelWrapper
from setup import (
    configure_hf_caches,
    setup_logging,
    extract_final_answer,
    normalize_answer,
    answers_match,
    reconstruct_prompt,
)


LOGGER = logging.getLogger("evaluate_steering")


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
        help="Identifier for a causal LM (default mirrors Step 1).",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        required=True,
        type=int,
        help="List of layer ids to use for steering (e.g., --layers 28).",
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
        default=Path("artifacts/steering_evaluation"),
        help="Directory for saving evaluation results.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of triples to evaluate (for debugging).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate.",
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


def load_steering_vectors(steering_dir: Path, layers: List[int]) -> Dict[int, np.ndarray]:
    """Load steering vectors for specified layers."""
    steering_vectors = {}
    for layer_id in layers:
        steering_path = steering_dir / f"layer{layer_id}_steering_vector.npy"
        if not steering_path.exists():
            LOGGER.warning("Steering vector not found for layer %d at %s", layer_id, steering_path)
            continue
        steering_vector = np.load(steering_path)
        steering_vectors[layer_id] = steering_vector
        LOGGER.info(
            "Loaded steering vector for layer %d (shape: %s, norm: %.4f)",
            layer_id,
            steering_vector.shape,
            np.linalg.norm(steering_vector),
        )
    return steering_vectors


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

    LOGGER.info("Loaded %d triples for evaluation", len(triples))

    # Load steering vectors
    steering_vectors = load_steering_vectors(args.steering_vectors_dir, args.layers)
    if not steering_vectors:
        LOGGER.error("No steering vectors loaded. Exiting.")
        return

    # Load model
    model = ModelWrapper(args.model_name, device=args.device)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluation results
    results = {
        "model_name": args.model_name,
        "layers": args.layers,
        "steering_coefficient": args.steering_coefficient,
        "num_triples": len(triples),
        "without_steering": {
            "correct": 0,
            "incorrect": 0,
            "accuracy": 0.0,
            "predictions": [],
        },
        "with_steering": {
            "correct": 0,
            "incorrect": 0,
            "accuracy": 0.0,
            "predictions": [],
        },
        "improvements": 0,  # Cases where steering fixed wrong predictions
        "degradations": 0,  # Cases where steering broke correct predictions
    }

    # Evaluate each triple
    for triple in tqdm(triples, desc="Evaluating steering"):
        sample_id = triple.get("sample_id", "unknown")
        prompt_text = triple.get("prompt", "")
        correct_answer = triple.get("correct_answer", "")
        metadata = triple.get("metadata", {})
        
        # Reconstruct prompt using template from metadata
        formatted_prompt = reconstruct_prompt(metadata, prompt_text)

        # Generate without steering
        try:
            response_no_steering = model.generate(
                formatted_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                strip_prompt=True,
                system_prompt=args.system_prompt,
            )
            predicted_no_steering = extract_final_answer(response_no_steering)
            correct_no_steering = answers_match(predicted_no_steering, correct_answer)
        except Exception as e:
            LOGGER.warning("Error generating without steering for sample %s: %s", sample_id, e)
            response_no_steering = ""
            predicted_no_steering = ""
            correct_no_steering = False

        # Generate with steering
        try:
            # Convert steering vectors to torch tensors (will be handled by generate_with_steering)
            steering_tensors = {layer_id: torch.from_numpy(vec) for layer_id, vec in steering_vectors.items()}
            
            response_with_steering = model.generate_with_steering(
                formatted_prompt,
                steering_vectors=steering_tensors,
                steering_coefficient=args.steering_coefficient,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                strip_prompt=True,
                system_prompt=args.system_prompt,
            )
            predicted_with_steering = extract_final_answer(response_with_steering)
            correct_with_steering = answers_match(predicted_with_steering, correct_answer)
        except Exception as e:
            LOGGER.warning("Error generating with steering for sample %s: %s", sample_id, e)
            response_with_steering = ""
            predicted_with_steering = ""
            correct_with_steering = False

        # Update results
        if correct_no_steering:
            results["without_steering"]["correct"] += 1
        else:
            results["without_steering"]["incorrect"] += 1

        if correct_with_steering:
            results["with_steering"]["correct"] += 1
        else:
            results["with_steering"]["incorrect"] += 1

        # Track improvements and degradations
        if not correct_no_steering and correct_with_steering:
            results["improvements"] += 1
        elif correct_no_steering and not correct_with_steering:
            results["degradations"] += 1

        # Store predictions
        results["without_steering"]["predictions"].append({
            "sample_id": sample_id,
            "predicted": predicted_no_steering,
            "correct": correct_answer,
            "is_correct": correct_no_steering,
            "response": response_no_steering[:500],  # Truncate for storage
        })
        results["with_steering"]["predictions"].append({
            "sample_id": sample_id,
            "predicted": predicted_with_steering,
            "correct": correct_answer,
            "is_correct": correct_with_steering,
            "response": response_with_steering[:500],  # Truncate for storage
        })

    # Calculate accuracies
    total = len(triples)
    results["without_steering"]["accuracy"] = results["without_steering"]["correct"] / total if total > 0 else 0.0
    results["with_steering"]["accuracy"] = results["with_steering"]["correct"] / total if total > 0 else 0.0

    # Save results
    results_path = args.output_dir / "steering_evaluation_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved evaluation results to %s", results_path)

    # Print summary
    LOGGER.info("=" * 60)
    LOGGER.info("Steering Evaluation Results")
    LOGGER.info("=" * 60)
    LOGGER.info("Total triples evaluated: %d", total)
    LOGGER.info("")
    LOGGER.info("Without steering:")
    LOGGER.info("  Correct: %d", results["without_steering"]["correct"])
    LOGGER.info("  Incorrect: %d", results["without_steering"]["incorrect"])
    LOGGER.info("  Accuracy: %.2f%%", results["without_steering"]["accuracy"] * 100)
    LOGGER.info("")
    LOGGER.info("With steering (coefficient=%.2f):", args.steering_coefficient)
    LOGGER.info("  Correct: %d", results["with_steering"]["correct"])
    LOGGER.info("  Incorrect: %d", results["with_steering"]["incorrect"])
    LOGGER.info("  Accuracy: %.2f%%", results["with_steering"]["accuracy"] * 100)
    LOGGER.info("")
    LOGGER.info("Improvements (wrong -> correct): %d", results["improvements"])
    LOGGER.info("Degradations (correct -> wrong): %d", results["degradations"])
    LOGGER.info("")
    accuracy_diff = results["with_steering"]["accuracy"] - results["without_steering"]["accuracy"]
    LOGGER.info("Accuracy change: %+.2f%%", accuracy_diff * 100)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()

