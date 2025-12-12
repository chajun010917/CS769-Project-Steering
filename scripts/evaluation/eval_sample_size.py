import os
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../<repo>/
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
TRIPLES = ARTIFACTS_DIR / "manual_review" / "10232025_human_review.json"
SYSTEM_PROMPT = "You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'."

OUTPUT_DIR = RESULTS_DIR / "sample_size_compare"
os.makedirs(OUTPUT_DIR, exist_ok=True)

configs = {
    "ss10":  ARTIFACTS_DIR / "steering_vectors_ss10",
    "ss50":  ARTIFACTS_DIR / "steering_vectors_ss50",
}

for name, vec_dir in configs.items():
    print(f"\n===== Evaluating sample size: {name} =====\n")

    cmd = [
        "python", "scripts/evaluate_steering.py",
        "--triples-path", TRIPLES,
        "--steering-vectors-dir", vec_dir,
        "--model-name", MODEL,
        "--layers", "28",
        "--steering-coefficient", "1.0",
        "--output-dir", OUTPUT_DIR,
        "--max-samples", "100",
        "--max-new-tokens", "1024",
        "--system-prompt", SYSTEM_PROMPT,
        "--sample-offset", "0",
    ]

    subprocess.run(cmd)

print("Done evaluating 10 & 50 sample sizes.")
