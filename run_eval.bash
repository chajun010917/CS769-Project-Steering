#!/usr/bin/env bash
set -euo pipefail

# End-to-end evaluation runner:
# 1) (Optional) compute missing steering vectors
# 2) alpha sweep evaluation -> results/alpha_sweep/alpha_*.json
# 3) plots: results/alpha_sweep/poster_alpha_CI.png, results/ablation/ablation_plot.png
# 4) (Optional) sample-size compare plot -> results/sample_size_compare/sample_size_vs_accuracy.png

# ====== CONFIG (override via env vars if desired) ======
: "${MODEL:=meta-llama/Llama-3.1-8B-Instruct}"
: "${LAYERS:=28}"

# Evaluation triples for alpha sweep
: "${TRIPLES_EVAL:=./artifacts/manual_review/12062025_human_review.json}"

# Training triples used to compute steering vectors (if missing)
: "${TRIPLES_TRAIN:=./artifacts/manual_review/12062025_human_review.json}"

: "${VECTORS_DIR:=artifacts/steering_vectors}"
: "${RESULTS_ALPHA_DIR:=results/alpha_sweep}"
: "${MAX_SAMPLES:=100}"
: "${SAMPLE_OFFSET:=0}"
: "${MAX_NEW_TOKENS:=1024}"
: "${TOKEN_SELECTION_METHOD:=last_token}" # last_token | gradient | dp_gradient | dp_average | token_mlp
: "${TOKEN_SELECTOR_MLP_PATH:=artifacts/mlp_models/token_selector.pt}"
: "${PROMPT:=You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'.}"

# Alpha sweep (must include 0.0 and 1.0 if you want ablation/sample-size plots)
ALPHAS=(0.0 0.2 0.5 0.8 1.0 1.2 1.5 2.0)

# Sample-size compare (optional; used by scripts/evaluation/plot_sample_size_compare.py)
: "${RUN_SAMPLE_SIZE_COMPARE:=1}"
: "${TRIPLES_SS_EVAL:=./artifacts/manual_review/10232025_human_review.json}"

mkdir -p "$RESULTS_ALPHA_DIR"

if [ ! -f "$TRIPLES_EVAL" ]; then
  echo "ERROR: Missing TRIPLES_EVAL at $TRIPLES_EVAL" >&2
  exit 1
fi

if [ ! -d "$VECTORS_DIR" ]; then
  mkdir -p "$VECTORS_DIR"
fi

echo "=== Step 0: Ensuring steering vectors exist in $VECTORS_DIR ==="
missing_any=0
for layer in $LAYERS; do
  if [ ! -f "$VECTORS_DIR/layer${layer}_steering_vector.npy" ]; then
    echo "Missing: $VECTORS_DIR/layer${layer}_steering_vector.npy"
    missing_any=1
  fi
done

if [ "$missing_any" -eq 1 ]; then
  if [ ! -f "$TRIPLES_TRAIN" ]; then
    echo "ERROR: Missing TRIPLES_TRAIN at $TRIPLES_TRAIN" >&2
    exit 1
  fi

  echo "Computing steering vectors..."
  token_selector_args=()
  if [ "$TOKEN_SELECTION_METHOD" = "token_mlp" ]; then
    if [ ! -f "$TOKEN_SELECTOR_MLP_PATH" ]; then
      echo "ERROR: TOKEN_SELECTION_METHOD=token_mlp but missing $TOKEN_SELECTOR_MLP_PATH" >&2
      exit 1
    fi
    token_selector_args=(--token-selection-mlp-path "$TOKEN_SELECTOR_MLP_PATH")
  fi

  python scripts/compute_steering_vectors.py \
    --triples-path "$TRIPLES_TRAIN" \
    --model-name "$MODEL" \
    --layers $LAYERS \
    --output-dir "$VECTORS_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --token-selection-method "$TOKEN_SELECTION_METHOD" \
    --system-prompt "$PROMPT" \
    "${token_selector_args[@]}"
else
  echo "All required steering vectors found."
fi

echo "=== Step 1: Alpha sweep evaluation -> $RESULTS_ALPHA_DIR ==="
for a in "${ALPHAS[@]}"; do
  echo -e "\n================ Running alpha = $a ================\n"

  # For alpha=0.0, do NOT skip baseline. For alpha>0, skip baseline for speed.
  if [ "$a" = "0.0" ]; then
    python scripts/evaluation/evaluate_steering_alpha_sweep.py \
      --triples-path "$TRIPLES_EVAL" \
      --model-name "$MODEL" \
      --layers $LAYERS \
      --steering-vectors-dir "$VECTORS_DIR" \
      --steering-coefficient "$a" \
      --max-samples "$MAX_SAMPLES" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --sample-offset "$SAMPLE_OFFSET" \
      --output-dir "$RESULTS_ALPHA_DIR" \
      --system-prompt "$PROMPT"
  else
    python scripts/evaluation/evaluate_steering_alpha_sweep.py \
      --triples-path "$TRIPLES_EVAL" \
      --model-name "$MODEL" \
      --layers $LAYERS \
      --steering-vectors-dir "$VECTORS_DIR" \
      --steering-coefficient "$a" \
      --max-samples "$MAX_SAMPLES" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --sample-offset "$SAMPLE_OFFSET" \
      --output-dir "$RESULTS_ALPHA_DIR" \
      --system-prompt "$PROMPT" \
      --skip-baseline
  fi
done

echo -e "\n===== ALL ALPHAS FINISHED ====="
echo "JSON results stored in: $RESULTS_ALPHA_DIR/"

echo "=== Step 2: Plot alpha sweep bootstrap CI ==="
python scripts/evaluation/plot_alpha_bootstrap.py

echo "=== Step 3: Ablation summary/plot (from sweep outputs) ==="
python scripts/evaluation/eval_alpha_sweep.py

if [ "$RUN_SAMPLE_SIZE_COMPARE" = "1" ]; then
  echo "=== Step 4: Sample-size compare (optional) ==="
  mkdir -p results/sample_size_compare

  if [ ! -f "$TRIPLES_SS_EVAL" ]; then
    echo "WARNING: Missing TRIPLES_SS_EVAL at $TRIPLES_SS_EVAL; skipping sample-size compare." >&2
  else
    for ss in 10 50; do
      vec_dir="artifacts/steering_vectors_ss${ss}"
      if [ ! -f "$vec_dir/layer28_steering_vector.npy" ]; then
        echo "Computing sample-size steering vectors (ss=$ss) -> $vec_dir"
        python scripts/compute_steering_vectors.py \
          --triples-path "$TRIPLES_SS_EVAL" \
          --model-name "$MODEL" \
          --layers 28 \
          --output-dir "$vec_dir" \
          --max-samples "$ss" \
          --token-selection-method last_token \
          --system-prompt "$PROMPT"
      fi

      echo "Evaluating vectors for ss=$ss"
      out_dir="results/sample_size_compare/ss${ss}"
      python scripts/evaluation/evaluate_steering.py \
        --triples-path "$TRIPLES_SS_EVAL" \
        --steering-vectors-dir "$vec_dir" \
        --model-name "$MODEL" \
        --layers 28 \
        --steering-coefficient 1.0 \
        --output-dir "$out_dir" \
        --max-samples 100 \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --system-prompt "$PROMPT" \
        --sample-offset 0

      # Copy into the filenames expected by scripts/evaluation/plot_sample_size_compare.py
      cp "$out_dir/steering_evaluation_results_offset0.json" "results/sample_size_compare/ss${ss}_steering_evaluation_results_offset0.json"
    done

    echo "=== Step 5: Plot sample-size comparison ==="
    python scripts/evaluation/plot_sample_size_compare.py
  fi
fi

