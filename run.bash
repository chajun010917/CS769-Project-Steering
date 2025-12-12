#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Gating Steering Experiment Runner
# ============================================================================
# This script runs the full pipeline for Gating Steering experiments.
#
# Usage:
#   ./run.bash
#
# Prerequisite:
#   Run ./setup.sh first to set up the environment.
# ============================================================================



# ---- Configuration Guide ----
# You can override these variables by setting them in your environment or editing below.
# Example: export POOLING_METHOD="mean" && ./run.bash
#
# Key Variables:
#   POOLING_METHOD: "per_token" (default), "mean", or "last_token"
#   TOKEN_SELECTION_METHOD: "last_token" (default), "gradient", "token_mlp"
#   MAX_SAMPLES: Number of samples to process (default: 100)
#   LAYERS: Space-separated list of layers to analyze (default: "26 27 28 29 30 31")
#   
# Skip Step Flags (set to 1 to skip):
#   SKIP_HIDDEN: Skip hidden state capture (Step 2)
#   SKIP_PROBES: Skip probe computation (Step 3)
#   SKIP_PLOTS: Skip visualization generation (Step 4)
#   SKIP_CRITICAL: Skip critical token analysis (Step 5)
#   SKIP_STEERING: Skip steering vector computation (Step 6)
#   SKIP_EVAL: Skip steering evaluation (Step 7)

# ---- Configuration ----
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MAX_SAMPLES=100
MAX_NEW_TOKENS=1024

# Pooling method: mean, last_token, or per_token
POOLING_METHOD="per_token"

# Layers to analyze (space-separated list)
# For Llama-3.1-8B: layers 0-31 available
# Focusing on layer 28 which shows good separation with mean pooling
LAYERS="26 27 28 29 30 31"

PROBE_MAX=1000  # Max samples per layer for probe data
DATASET_PATH="UW-Madison-Lee-Lab/MMLU-Pro-CoT-Eval"

# Hugging Face caches (define before using HF_HOME in paths)
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

# Derived artifact locations
ANALYSIS_OUTPUT="${HF_HOME}/artifacts/probe_analysis"
PLOT_OUTPUT="${HF_HOME}/reports/hidden_state_viz_${POOLING_METHOD}"

# Step skip controls (set to 1 to skip)
SKIP_HIDDEN="${SKIP_HIDDEN:-0}"
SKIP_PROBES="${SKIP_PROBES:-0}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
SKIP_CRITICAL="${SKIP_CRITICAL:-0}"
SKIP_STEERING="${SKIP_STEERING:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_EMBED="${SKIP_EMBED:-0}"

TOKEN_SELECTION_METHOD="last_token"  # last_token | gradient | dp_gradient | dp_average | token_mlp
LAYER_SELECTION_METHOD="fixed"  

DP_ALIGNMENT_ARGS=(--dp-alignment)

SYSTEM_PROMPT="You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'."

# ---- Setup Verification ----
# Ensure HF directories exist
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

# ---- step 1: use triple set ----
TRIPLES_PATH="./artifacts/manual_review/12062025_human_review.json"
TRAIN_OFFSET=0
EVAL_OFFSET=100

if [ -f "${TRIPLES_PATH}" ]; then
  echo "=== Step 1: Using manually reviewed triples: ${TRIPLES_PATH} ==="
else
  echo "ERROR: Manual triples file not found at ${TRIPLES_PATH}"
  echo "Please check the path or ensure the file exists."
  exit 1
fi

# ---- step 2: capture hidden states ----
# This step generates the probe data (.npz) files needed for Step 3.
if [ "${SKIP_HIDDEN}" = "1" ]; then
    echo "=== Step 2: Skipping hidden state capture (SKIP_HIDDEN=1) ==="
else
    echo "=== Step 2: Capturing hidden states with ${POOLING_METHOD} pooling ==="
    
    python scripts/collect_hidden_states.py \
      --triples-path "${TRIPLES_PATH}" \
      --model-name "${MODEL_ID}" \
      --layers ${LAYERS} \
      --probe-max-samples "${PROBE_MAX}" \
      --max-samples "${MAX_SAMPLES}" \
      --pooling-method "${POOLING_METHOD}" \
      --alignment-layer 28 \
      --dp-max-shift 40 \
      --system-prompt "${SYSTEM_PROMPT}" \
      "${DP_ALIGNMENT_ARGS[@]}"
fi

# ---- step 3: compute probes and vectors for multiple layers ----
if [ "${SKIP_PROBES}" = "1" ]; then
  echo "=== Step 3: Skipping probe computation (SKIP_PROBES=1) ==="
else
  echo "=== Step 3: Computing probes and vectors ==="
  ANALYSIS_OUTPUT="${HF_HOME}/artifacts/probe_analysis"

  for layer in ${LAYERS}; do
    PROBE_DATA="${HF_HOME}/artifacts/probe_data/layer${layer}_probe_data.npz"

    if [ -f "${PROBE_DATA}" ]; then
      echo "  Computing probes for layer ${layer}..."
      cmd=(
        python scripts/compute_probes.py
        --probe-data-path "${PROBE_DATA}"
        --output-dir "${ANALYSIS_OUTPUT}"
        --alignments-dir "${HF_HOME}/artifacts/alignments"
        --seed 42
      )
      cmd+=("${DP_ALIGNMENT_ARGS[@]}")
      "${cmd[@]}"
    else
      echo "  Warning: Probe data not found for layer ${layer} at ${PROBE_DATA}"
      echo "  Ensure Step 2 ran successfully."
    fi
  done
fi

# ---- step 4: generate visualizations for multiple layers ----
if [ "${SKIP_PLOTS}" = "1" ]; then
  echo "=== Step 4: Skipping visualizations (SKIP_PLOTS=1) ==="
else
  if [ -d "${ANALYSIS_OUTPUT}" ]; then
    echo "=== Step 4: Generating visualizations ==="
    for layer in ${LAYERS}; do
      METRICS_FILE="${ANALYSIS_OUTPUT}/layer${layer}_metrics.json"
      if [ -f "${METRICS_FILE}" ]; then
        echo "  Plotting layer ${layer}..."
        cmd=(
          python scripts/plot_probes.py
          --analysis-dir "${ANALYSIS_OUTPUT}"
          --layer ${layer}
          --output-dir "${PLOT_OUTPUT}"
        )
        cmd+=("${DP_ALIGNMENT_ARGS[@]}")
        "${cmd[@]}"
      else
        echo "  Warning: Metrics not found for layer ${layer}, skipping plots"
      fi
    done
  else
    echo "=== Step 4: Skipping visualizations (analysis directory missing) ==="
  fi
fi

# ---- step 5: analyze critical tokens (only for per_token pooling) ----
if [ "${SKIP_CRITICAL}" = "1" ]; then
  echo "=== Step 5: Skipping critical token analysis (SKIP_CRITICAL=1) ==="
elif [ "${POOLING_METHOD}" = "per_token" ]; then
  echo "=== Step 5: Analyzing critical token positions ==="
  CRITICAL_TOKENS_OUTPUT="${HF_HOME}/reports/critical_tokens_${POOLING_METHOD}"

  for layer in ${LAYERS}; do
    PROBE_DATA="${HF_HOME}/artifacts/probe_data/layer${layer}_probe_data.npz"

    if [ -f "${PROBE_DATA}" ]; then
      echo "  Analyzing critical tokens for layer ${layer}..."
      cmd=(
        python scripts/analyze_critical_tokens.py
        --probe-data-path "${PROBE_DATA}"
        --alignments-dir "${HF_HOME}/artifacts/alignments"
        --output-dir "${CRITICAL_TOKENS_OUTPUT}"
        --top-k 20
      )
      cmd+=("${DP_ALIGNMENT_ARGS[@]}")
      "${cmd[@]}"
    else
      echo "  Warning: Probe data not found for layer ${layer}"
    fi
  done
else
  echo "=== Step 5: Skipping critical token analysis (only applicable for per_token pooling) ==="
fi

# ---- step 6: compute steering vectors ----
STEERING_VECTORS_DIR="${HF_HOME}/artifacts/steering_vectors"
if [ "${SKIP_STEERING}" = "1" ]; then
  echo "=== Step 6: Skipping steering vector computation (SKIP_STEERING=1) ==="
else
  echo "=== Step 6: Computing steering vectors from last_token representations ==="
  
  TOKEN_SELECTOR_ARGS=()
  if [ "${TOKEN_SELECTION_METHOD}" = "token_mlp" ]; then
    TOKEN_SELECTOR_ARGS+=(--token-selection-mlp-path "${TOKEN_SELECTOR_MLP_PATH}")
  fi

  python scripts/compute_steering_vectors.py \
    --triples-path "${TRIPLES_PATH}" \
    --model-name "${MODEL_ID}" \
    --layers ${LAYERS} \
    --output-dir "${STEERING_VECTORS_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    --token-selection-method "${TOKEN_SELECTION_METHOD}" \
    --alignments-dir "${HF_HOME}/artifacts/alignments" \
    --system-prompt "${SYSTEM_PROMPT}" \
    "${TOKEN_SELECTOR_ARGS[@]}"
fi

# ---- step 7: evaluate steering ----
if [ "${SKIP_EVAL}" = "1" ]; then
  echo "=== Step 7: Skipping steering evaluation (SKIP_EVAL=1) ==="
else
echo "=== Step 7: Evaluating steering vectors ==="
STEERING_EVAL_DIR_IN="${HF_HOME}/artifacts/steering_evaluation_in_sample"
STEERING_EVAL_DIR_OUT="${HF_HOME}/artifacts/steering_evaluation_out_of_sample"

# Only evaluate steering if vectors were computed
if [ -d "${STEERING_VECTORS_DIR}" ]; then
  # Evaluate each layer separately (or all together - you can modify this)
  # For now, evaluate with the best layer (typically layer 28)
  # To test a specific layer (e.g., layer 26), change BEST_LAYER="26"
  BEST_LAYER="29"  # Can be changed based on visualization results  # can add a list of layers here
  EVAL_LAYERS="${BEST_LAYER}"
  
  # Baseline results (known: 8 correct, 92 incorrect for 100 samples)
  # Set SKIP_BASELINE=1 to skip baseline computation and use provided values
  SKIP_BASELINE=0
  BASELINE_CORRECT=0
  BASELINE_INCORRECT=0
  
  # Check if steering vector exists for the best layer
  if [ -f "${STEERING_VECTORS_DIR}/layer${BEST_LAYER}_steering_vector.npy" ]; then
    echo "Evaluating steering for layer ${BEST_LAYER}..."
    
    if [ "${SKIP_BASELINE}" = "1" ]; then
      echo "Skipping baseline evaluation (using provided: ${BASELINE_CORRECT} correct, ${BASELINE_INCORRECT} incorrect)"
      SKIP_ARGS=(--skip-baseline --baseline-correct "${BASELINE_CORRECT}" --baseline-incorrect "${BASELINE_INCORRECT}")
    else
      SKIP_ARGS=()
    fi

    echo "Evaluating in-sample steering performance..."
    python scripts/evaluate_steering.py \
      --triples-path "${TRIPLES_PATH}" \
      --steering-vectors-dir "${STEERING_VECTORS_DIR}" \
      --model-name "${MODEL_ID}" \
      --layers ${EVAL_LAYERS} \
      --steering-coefficient 1.0 \
      --output-dir "${STEERING_EVAL_DIR_IN}" \
      --hidden-states-dir "${HF_HOME}/artifacts/hidden_states" \
      --max-samples "${MAX_SAMPLES}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --system-prompt "${SYSTEM_PROMPT}" \
      --sample-offset "${TRAIN_OFFSET}" \
      "${SKIP_ARGS[@]}" \

    echo "Evaluating out-of-sample steering performance..."
    python scripts/evaluate_steering.py \
      --triples-path "${TRIPLES_PATH}" \
      --steering-vectors-dir "${STEERING_VECTORS_DIR}" \
      --model-name "${MODEL_ID}" \
      --layers ${EVAL_LAYERS} \
      --steering-coefficient 1.0 \
      --output-dir "${STEERING_EVAL_DIR_OUT}" \
      --hidden-states-dir "${HF_HOME}/artifacts/hidden_states" \
      --max-samples "${MAX_SAMPLES}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --system-prompt "${SYSTEM_PROMPT}" \
      --sample-offset "${EVAL_OFFSET}" \
      "${SKIP_ARGS[@]}" \
    
    echo "Steering evaluation complete. Results saved to ${STEERING_EVAL_DIR_IN} and ${STEERING_EVAL_DIR_OUT}"
  else
    echo "  Warning: Steering vector not found for layer ${BEST_LAYER}, skipping evaluation"
  fi
else
  echo "=== Step 7: Skipping steering evaluation (steering vectors not computed) ==="
fi
fi

# ---- step 8: compare embeddings with and without steering ----
if [ "${SKIP_EMBED}" = "1" ]; then
  echo "=== Step 8: Skipping embedding comparison (SKIP_EMBED=1) ==="
else
  echo "=== Step 8: Comparing embeddings with and without steering ==="
  EMBEDDING_COMPARISON_DIR_IN="${HF_HOME}/reports/steering_embedding_comparison_in_sample"
  EMBEDDING_COMPARISON_DIR_OUT="${HF_HOME}/reports/steering_embedding_comparison_out_of_sample"

  # Only compare embeddings if steering vectors exist
  if [ -d "${STEERING_VECTORS_DIR}" ]; then
    if [ -f "${STEERING_VECTORS_DIR}/layer${BEST_LAYER}_steering_vector.npy" ]; then
      echo "Comparing embeddings for layer ${BEST_LAYER}..."
      python scripts/compare_steering_embeddings.py \
        --triples-path "${TRIPLES_PATH}" \
        --steering-vectors-dir "${STEERING_VECTORS_DIR}" \
        --model-name "${MODEL_ID}" \
        --layer ${BEST_LAYER} \
        --steering-coefficient 1.0 \
        --output-dir "${EMBEDDING_COMPARISON_DIR_IN}" \
        --max-samples "${MAX_SAMPLES}" \
        --system-prompt "You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'." \
        --pooling-method "${POOLING_METHOD}" \
        --sample-offset "${TRAIN_OFFSET}" \
        --alignments-dir "${HF_HOME}/artifacts/alignments" \
        --seed 42
      
      python scripts/compare_steering_embeddings.py \
        --triples-path "${TRIPLES_PATH}" \
        --steering-vectors-dir "${STEERING_VECTORS_DIR}" \
        --model-name "${MODEL_ID}" \
        --layer ${BEST_LAYER} \
        --steering-coefficient 1.0 \
        --output-dir "${EMBEDDING_COMPARISON_DIR_OUT}" \
        --max-samples "${MAX_SAMPLES}" \
        --system-prompt "You are a careful reasoning assistant. Think step by step and end with 'Final answer: <choice>'." \
        --pooling-method "${POOLING_METHOD}" \
        --sample-offset "${EVAL_OFFSET}" \
        --alignments-dir "${HF_HOME}/artifacts/alignments" \
        --seed 42
      
      echo "Embedding comparison complete. Results saved to ${EMBEDDING_COMPARISON_DIR_IN} and ${EMBEDDING_COMPARISON_DIR_OUT}"
    else
      echo "  Warning: Steering vector not found for layer ${BEST_LAYER}, skipping embedding comparison"
    fi
  else
    echo "=== Step 8: Skipping embedding comparison (steering vectors not computed) ==="
  fi
fi

echo ""
echo "=== Done! ==="
echo "Results:"
echo "  - Triples source: ${TRIPLES_PATH}"
echo "  - Train offset: ${TRAIN_OFFSET}, Eval offset: ${EVAL_OFFSET}"
echo "  - Hidden states: ${HF_HOME}/artifacts/hidden_states/"
echo "  - Alignments: ${HF_HOME}/artifacts/alignments/"
echo "  - Probe data: ${HF_HOME}/artifacts/probe_data/"
echo "  - Computed probes: ${ANALYSIS_OUTPUT}/"
echo "  - Visualizations: ${PLOT_OUTPUT}/"
if [ -d "${STEERING_VECTORS_DIR}" ]; then
  echo "  - Steering vectors: ${STEERING_VECTORS_DIR}/"
  echo "  - Steering evaluation (in-sample): ${STEERING_EVAL_DIR_IN}/"
  echo "  - Steering evaluation (out-of-sample): ${STEERING_EVAL_DIR_OUT}/"
  echo "  - Embedding comparison (in-sample): ${EMBEDDING_COMPARISON_DIR_IN}/"
  echo "  - Embedding comparison (out-of-sample): ${EMBEDDING_COMPARISON_DIR_OUT}/"
fi
echo ""
echo "Processed layers: ${LAYERS}"