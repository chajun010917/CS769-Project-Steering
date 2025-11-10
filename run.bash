#!/usr/bin/env bash
set -euo pipefail

# ---- Initialize conda ----
# Source conda.sh to enable conda commands in this script
# if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
#     source "$HOME/anaconda3/etc/profile.d/conda.sh"
# else
#     echo "Error: conda.sh not found. Please ensure Anaconda/Miniconda is installed."
#     exit 1
# fi

# ---- config for the smoke test ----
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MAX_SAMPLES=100
MAX_NEW_TOKENS=1024

# Layers to analyze (space-separated list)
# For Llama-3.1-8B: layers 0-31 available
# Focusing on layer 28 which shows good separation with mean pooling
LAYERS="26 27 28 29 30 31"

# Pooling method: mean, last_token, or per_token
POOLING_METHOD="last_token"

PROBE_MAX=1000  # Max samples per layer for probe data
DATASET_PATH="UW-Madison-Lee-Lab/MMLU-Pro-CoT-Eval"
DATASET_CONFIG="ALL"

# ---- setup ----

# Create environment if it doesn't exist
# if ! conda info --envs | grep -q '^steering'; then
#   conda create -y -n steering python=3.10
# fi
# conda activate steering
export HF_HOME="$(pwd)/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="$(pwd)/.cache/transformers"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

# echo "Installing requirements..."
# pip install -q --upgrade pip
# pip3 install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
# pip install -q -r requirements.txt

# If first time using HF Hub, uncomment and run once (will prompt for token)
#hf auth login

# ---- step 1: generate triple set ----
TRIPLES_OUT="artifacts/triples/triples_small.jsonl"

if [ -f "${TRIPLES_OUT}" ]; then
  echo "=== Step 1: Skipping triple generation (file exists: ${TRIPLES_OUT}) ==="
  echo "To regenerate, delete the file and rerun."
else
  echo "=== Step 1: Generating triples ==="
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python scripts/prepare_triples.py \
    --dataset-name "${DATASET_PATH}" \
    --split test \
    --model-name "${MODEL_ID}" \
    --max-samples "${MAX_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --only-wrong \
    --output-path "${TRIPLES_OUT}"
fi

# ---- step 2: capture hidden states ----
# echo "=== Step 2: Capturing hidden states with ${POOLING_METHOD} pooling ==="
# # Note: collect_hidden_states.py will generate probe data for all layers specified
# python scripts/collect_hidden_states.py \
#   --triples-path "${TRIPLES_OUT}" \
#   --model-name "${MODEL_ID}" \
#   --layers ${LAYERS} \
#   --probe-max-samples "${PROBE_MAX}" \
#   --max-samples "${MAX_SAMPLES}" \
#   --pooling-method "${POOLING_METHOD}"

# echo "Generated probe data for layers: ${LAYERS}"

# ---- step 3: compute probes and vectors for multiple layers ----
# echo "=== Step 3: Computing probes and vectors ==="
# ANALYSIS_OUTPUT="artifacts/probe_analysis"

# # Process each layer from the LAYERS variable
# for layer in ${LAYERS}; do
#   PROBE_DATA="artifacts/probe_data/layer${layer}_probe_data.npz"
  
#   if [ -f "${PROBE_DATA}" ]; then
#     echo "  Computing probes for layer ${layer}..."
#     python scripts/compute_probes.py \
#       --probe-data-path "${PROBE_DATA}" \
#       --output-dir "${ANALYSIS_OUTPUT}" \
#       --seed 42
#   else
#     echo "  Warning: Probe data not found for layer ${layer} at ${PROBE_DATA}"
#   fi
# done

# ---- step 4: generate visualizations for multiple layers ----
# echo "=== Step 4: Generating visualizations ==="
# PLOT_OUTPUT="reports/hidden_state_viz_${POOLING_METHOD}"

# # Plot each layer that was computed
# for layer in ${LAYERS}; do
#   METRICS_FILE="${ANALYSIS_OUTPUT}/layer${layer}_metrics.json"
  
#   if [ -f "${METRICS_FILE}" ]; then
#     echo "  Plotting layer ${layer}..."
#     python scripts/plot_probes.py \
#       --analysis-dir "${ANALYSIS_OUTPUT}" \
#       --layer ${layer} \
#       --output-dir "${PLOT_OUTPUT}"
#   else
#     echo "  Warning: Metrics not found for layer ${layer}, skipping plots"
#   fi
# done

# ---- step 5: analyze critical tokens (only for per_token pooling) ----
if [ "${POOLING_METHOD}" = "per_token" ]; then
  echo "=== Step 5: Analyzing critical token positions ==="
  CRITICAL_TOKENS_OUTPUT="reports/critical_tokens_${POOLING_METHOD}"

  for layer in ${LAYERS}; do
    PROBE_DATA="artifacts/probe_data/layer${layer}_probe_data.npz"
    
    if [ -f "${PROBE_DATA}" ]; then
      echo "  Analyzing critical tokens for layer ${layer}..."
      python scripts/analyze_critical_tokens.py \
        --probe-data-path "${PROBE_DATA}" \
        --alignments-dir "artifacts/alignments" \
        --output-dir "${CRITICAL_TOKENS_OUTPUT}" \
        --top-k 20
    else
      echo "  Warning: Probe data not found for layer ${layer}"
    fi
  done
else
  echo "=== Step 5: Skipping critical token analysis (only applicable for per_token pooling) ==="
fi

# ---- step 6: compute steering vectors ----
echo "=== Step 6: Computing steering vectors from last_token representations ==="
STEERING_VECTORS_DIR="artifacts/steering_vectors"

# Only compute steering vectors if using last_token pooling
# if [ "${POOLING_METHOD}" = "last_token" ]; then
#   echo "Computing steering vectors for layers: ${LAYERS}"
#   python scripts/compute_steering_vectors.py \
#     --triples-path "${TRIPLES_OUT}" \
#     --model-name "${MODEL_ID}" \
#     --layers ${LAYERS} \
#     --output-dir "${STEERING_VECTORS_DIR}" \
#     --max-samples "${MAX_SAMPLES}"
  
#   echo "Steering vectors computed and saved to ${STEERING_VECTORS_DIR}"
# else
#   echo "=== Step 6: Skipping steering vector computation (only applicable for last_token pooling) ==="
# fi

# ---- step 7: evaluate steering ----
echo "=== Step 7: Evaluating steering vectors ==="
STEERING_EVAL_DIR="artifacts/steering_evaluation"

# Only evaluate steering if vectors were computed
if [ "${POOLING_METHOD}" = "last_token" ] && [ -d "${STEERING_VECTORS_DIR}" ]; then
  # Evaluate each layer separately (or all together - you can modify this)
  # For now, evaluate with the best layer (typically layer 28)
  # To test a specific layer (e.g., layer 26), change BEST_LAYER="26"
  BEST_LAYER="29"  # Can be changed based on visualization results  # can add a list of layers here
  
  # Baseline results (known: 8 correct, 92 incorrect for 100 samples)
  # Set SKIP_BASELINE=1 to skip baseline computation and use provided values
  SKIP_BASELINE=1
  BASELINE_CORRECT=8
  BASELINE_INCORRECT=92
  
  # Check if steering vector exists for the best layer
  if [ -f "${STEERING_VECTORS_DIR}/layer${BEST_LAYER}_steering_vector.npy" ]; then
    echo "Evaluating steering for layer ${BEST_LAYER}..."
    
    if [ "${SKIP_BASELINE}" = "1" ]; then
      echo "Skipping baseline evaluation (using provided: ${BASELINE_CORRECT} correct, ${BASELINE_INCORRECT} incorrect)"
      python scripts/evaluate_steering.py \
        --triples-path "${TRIPLES_OUT}" \
        --steering-vectors-dir "${STEERING_VECTORS_DIR}" \
        --model-name "${MODEL_ID}" \
        --layers ${BEST_LAYER} \
        --steering-coefficient 1.0 \
        --output-dir "${STEERING_EVAL_DIR}" \
        --max-samples "${MAX_SAMPLES}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --system-prompt "You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'." \
        --skip-baseline \
        --baseline-correct "${BASELINE_CORRECT}" \
        --baseline-incorrect "${BASELINE_INCORRECT}"
    else
      echo "Evaluating with baseline computation..."
      python scripts/evaluate_steering.py \
        --triples-path "${TRIPLES_OUT}" \
        --steering-vectors-dir "${STEERING_VECTORS_DIR}" \
        --model-name "${MODEL_ID}" \
        --layers ${BEST_LAYER} \
        --steering-coefficient 1.0 \
        --output-dir "${STEERING_EVAL_DIR}" \
        --max-samples "${MAX_SAMPLES}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --system-prompt "You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'."
    fi
    
    echo "Steering evaluation complete. Results saved to ${STEERING_EVAL_DIR}"
  else
    echo "  Warning: Steering vector not found for layer ${BEST_LAYER}, skipping evaluation"
  fi
else
  echo "=== Step 7: Skipping steering evaluation (steering vectors not computed) ==="
fi

# ---- step 8: compare embeddings with and without steering ----
echo "=== Step 8: Comparing embeddings with and without steering ==="
EMBEDDING_COMPARISON_DIR="reports/steering_embedding_comparison"

# Only compare embeddings if steering vectors exist
if [ "${POOLING_METHOD}" = "last_token" ] && [ -d "${STEERING_VECTORS_DIR}" ]; then
  if [ -f "${STEERING_VECTORS_DIR}/layer${BEST_LAYER}_steering_vector.npy" ]; then
    echo "Comparing embeddings for layer ${BEST_LAYER}..."
    python scripts/compare_steering_embeddings.py \
      --triples-path "${TRIPLES_OUT}" \
      --steering-vectors-dir "${STEERING_VECTORS_DIR}" \
      --model-name "${MODEL_ID}" \
      --layer ${BEST_LAYER} \
      --steering-coefficient 1.0 \
      --output-dir "${EMBEDDING_COMPARISON_DIR}" \
      --max-samples "${MAX_SAMPLES}" \
      --system-prompt "You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'." \
      --pooling-method "${POOLING_METHOD}" \
      --seed 42
    
    echo "Embedding comparison complete. Results saved to ${EMBEDDING_COMPARISON_DIR}"
  else
    echo "  Warning: Steering vector not found for layer ${BEST_LAYER}, skipping embedding comparison"
  fi
else
  echo "=== Step 8: Skipping embedding comparison (steering vectors not computed) ==="
fi

echo ""
echo "=== Done! ==="
echo "Results:"
echo "  - Triples: ${TRIPLES_OUT}"
echo "  - Hidden states: artifacts/hidden_states/"
echo "  - Alignments: artifacts/alignments/"
echo "  - Probe data: artifacts/probe_data/"
echo "  - Computed probes: ${ANALYSIS_OUTPUT}/"
echo "  - Visualizations: ${PLOT_OUTPUT}/"
if [ "${POOLING_METHOD}" = "last_token" ]; then
  echo "  - Steering vectors: ${STEERING_VECTORS_DIR}/"
  echo "  - Steering evaluation: ${STEERING_EVAL_DIR}/"
  echo "  - Embedding comparison: ${EMBEDDING_COMPARISON_DIR}/"
fi
echo ""
echo "Processed layers: ${LAYERS}"