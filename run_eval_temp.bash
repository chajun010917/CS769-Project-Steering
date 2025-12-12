#!/bin/bash

# ====== CONFIG ======
TRIPLES="./artifacts/manual_review/12062025_human_review.json"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
LAYERS="28"
OUTDIR="results/alpha_sweeptest"
VECTORS="artifacts/steering_vectors"
MAX=100
OFFSET=0
TOKENS=1024
PROMPT="You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'."

# ====== ALPHAS TO SWEEP ======
#ALPHAS=(0.0 0.2 0.5 0.8 1.0 1.2 1.5 2)
ALPHAS=(0.0 0.2)

mkdir -p "$OUTDIR"

for a in "${ALPHAS[@]}"; do
    echo -e "\n================ Running alpha = $a ================\n"

    # ⚠️ Only for alpha=0: DO NOT skip baseline
    if [ "$a" == "0.0" ]; then
        python scripts/evaluation/evaluate_steering_alpha_sweep.py \
            --triples-path "$TRIPLES" \
            --model-name "$MODEL" \
            --layers $LAYERS \
            --steering-vectors-dir "$VECTORS" \
            --steering-coefficient $a \
            --max-samples $MAX \
            --max-new-tokens $TOKENS \
            --sample-offset $OFFSET \
            --output-dir "$OUTDIR" \
            --system-prompt "$PROMPT"
    else
        python scripts/evaluation/evaluate_steering_alpha_sweep.py \
            --triples-path "$TRIPLES" \
            --model-name "$MODEL" \
            --layers $LAYERS \
            --steering-vectors-dir "$VECTORS" \
            --steering-coefficient $a \
            --max-samples $MAX \
            --max-new-tokens $TOKENS \
            --sample-offset $OFFSET \
            --output-dir "$OUTDIR" \
            --system-prompt "$PROMPT" \
            --skip-baseline
    fi
done

echo -e "\n===== ALL ALPHAS FINISHED ====="
echo "JSON results stored in: $OUTDIR/"

CUDA_VISIBLE_DEVICES="" python scripts/compute_steering_vectors.py \
  --triples-path artifacts/manual_review/10232025_human_review.json \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --layers 28 \
  --output-dir artifacts/steering_vectors_ss10 \
  --max-samples 10 \
  --token-selection-method last_token \
  --system-prompt "You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'."

CUDA_VISIBLE_DEVICES="" python scripts/compute_steering_vectors.py \
  --triples-path artifacts/manual_review/10232025_human_review.json \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --layers 28 \
  --output-dir artifacts/steering_vectors_ss50 \
  --max-samples 50 \
  --token-selection-method last_token \
  --system-prompt "You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'."

