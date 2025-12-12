#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Experiment Runner Script
# ============================================================================
# This script runs multiple experiments with different parameter combinations.
#
# Usage:
#   ./run_experiments.sh
#   ./run_experiments.sh --dry-run  # Preview experiments without running
# ============================================================================

# ---- Configuration ----
EXPERIMENTS_DIR="experiments"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_BASE_DIR="${EXPERIMENTS_DIR}/${TIMESTAMP}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_BASH="${SCRIPT_DIR}/run.bash"

if [ ! -f "${RUN_BASH}" ]; then
    echo "Error: run.bash not found at ${RUN_BASH}"
    exit 1
fi

# Dry run flag
DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE: Previewing experiments ==="
fi

# ---- Experiment Configurations ----
# Define parameter combinations to test.
# Format: "PARAM1=value1 PARAM2=value2 ..."
declare -a EXPERIMENTS=(
    # Baseline: per_token pooling, last_token selection
    "POOLING_METHOD=per_token TOKEN_SELECTION_METHOD=last_token LAYER_SELECTION_METHOD=fixed SKIP_HIDDEN=0 SKIP_PROBES=0 SKIP_PLOTS=0 SKIP_CRITICAL=0 SKIP_STEERING=0 SKIP_EVAL=0"
    
    # Variation: gradient-based selection
    # "POOLING_METHOD=per_token TOKEN_SELECTION_METHOD=gradient LAYER_SELECTION_METHOD=fixed SKIP_HIDDEN=1 SKIP_PROBES=1 SKIP_PLOTS=0 SKIP_CRITICAL=1 SKIP_STEERING=0 SKIP_EVAL=0"
)

# ---- Helper Functions ----

generate_experiment_name() {
    local params="$1"
    local name=""
    for param in $params; do
        local key="${param%%=*}"
        local value="${param#*=}"
        if [ -z "$name" ]; then
            name="${key}_${value}"
        else
            name="${name}__${key}_${value}"
        fi
    done
    echo "$name"
}

create_modified_run_script() {
    local exp_params="$1"
    local output_script="$2"
    
    # Copy original script
    cp "${RUN_BASH}" "${output_script}"
    
    # Override parameters
    for param in $exp_params; do
        local key="${param%%=*}"
        local value="${param#*=}"
        
        # Escape special characters in the key for sed
        local escaped_key=$(printf '%s\n' "$key" | sed 's/[[\.*^$()+?{|]/\\&/g')
        
        # Replace assignment in script
        # Supports KEY="value", KEY=value, or KEY="${KEY:-default}" (via matching start of line)
        local temp_file=$(mktemp)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed "s|^${escaped_key}=.*|${key}=\"${value}\"|" "${output_script}" > "${temp_file}"
        else
            sed "s|^${escaped_key}=.*|${key}=\"${value}\"|" "${output_script}" > "${temp_file}"
        fi
        mv "${temp_file}" "${output_script}"
    done
    
    # Explicitly force SKIP_EMBED=0 to always run embedding comparison
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|^SKIP_EMBED=.*|SKIP_EMBED=0|" "${output_script}"
    else
        sed -i "s|^SKIP_EMBED=.*|SKIP_EMBED=0|" "${output_script}"
    fi

    # Handle DP_ALIGNMENT_ARGS logic based on method
    local token_method="last_token"
    local pooling_method="per_token"
    
    for param in $exp_params; do
        if [[ "${param}" == TOKEN_SELECTION_METHOD=* ]]; then token_method="${param#*=}"; fi
        if [[ "${param}" == POOLING_METHOD=* ]]; then pooling_method="${param#*=}"; fi
    done
    
    local needs_dp=false
    if [[ "$token_method" == "dp_gradient" ]] || [[ "$token_method" == "dp_average" ]] || [[ "$pooling_method" == "per_token" ]]; then
        needs_dp=true
    fi
    
    if [ "$needs_dp" = true ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|^DP_ALIGNMENT_ARGS=.*|DP_ALIGNMENT_ARGS=(--dp-alignment)|" "${output_script}"
        else
            sed -i "s|^DP_ALIGNMENT_ARGS=.*|DP_ALIGNMENT_ARGS=(--dp-alignment)|" "${output_script}"
        fi
    else
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|^DP_ALIGNMENT_ARGS=.*|DP_ALIGNMENT_ARGS=()|" "${output_script}"
        else
            sed -i "s|^DP_ALIGNMENT_ARGS=.*|DP_ALIGNMENT_ARGS=()|" "${output_script}"
        fi
    fi
    
    chmod +x "${output_script}"
}

run_experiment() {
    local exp_params="$1"
    local exp_name=$(generate_experiment_name "$exp_params")
    local exp_dir="${EXPERIMENT_BASE_DIR}/${exp_name}"
    local log_file="${exp_dir}/run.log"
    local summary_file="${exp_dir}/summary.txt"
    
    echo "Running experiment: ${exp_name}"
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would create: ${exp_dir}"
        return 0
    fi
    
    mkdir -p "${exp_dir}/images"
    local modified_script="${exp_dir}/run_modified.sh"
    create_modified_run_script "$exp_params" "$modified_script"
    
    {
        echo "Experiment: ${exp_name}"
        echo "Timestamp: $(date)"
        echo "Parameters: ${exp_params}"
    } > "${summary_file}"
    
    # Run the experiment
    local script_dir=$(dirname "${RUN_BASH}")
    local start_time=$(date +%s)
    
    # Execute from script directory to resolve relative paths
    if (cd "${script_dir}" && /bin/bash "${modified_script}") 2>&1 | tee "${log_file}"; then
        echo "Status: SUCCESS" >> "${summary_file}"
        echo "✓ Experiment completed successfully"
        copy_experiment_images "$exp_params" "$exp_dir" "$script_dir" "$EXPERIMENT_BASE_DIR" "$exp_name"
    else
        echo "Status: FAILED" >> "${summary_file}"
        echo "✗ Experiment failed (check run.log)"
        copy_experiment_images "$exp_params" "$exp_dir" "$script_dir" "$EXPERIMENT_BASE_DIR" "$exp_name"
    fi
}

copy_experiment_images() {
    local exp_params="$1"
    local exp_dir="$2"
    local script_dir="$3"
    local experiment_base_dir="$4"
    local exp_name="$5"
    
    # Simple extraction of HF_HOME for image copying
    # (Simplified from original script for brevity, assumes default or exported var)
    local hf_home="${HF_HOME:-/nobackup/bdeka/huggingface_cache}"
    
    # Try to find meaningful HF_HOME from log if custom
    if [ -f "${exp_dir}/run.log" ]; then
        local log_hf=$(grep "HF_HOME=" "${exp_dir}/run.log" | head -1 | cut -d= -f2 | tr -d '"')
        if [ -n "$log_hf" ]; then hf_home="$log_hf"; fi
    fi
    
    local images_dir="${exp_dir}/images"
    local pooling_method="per_token"
    if [[ "$exp_params" == *"POOLING_METHOD="* ]]; then
        pooling_method=$(echo "$exp_params" | grep -o "POOLING_METHOD=[^ ]*" | cut -d= -f2)
    fi
    
    # Copy from known report locations
    local reports_dir="${hf_home}/reports"
    if [ -d "$reports_dir" ]; then
        # Find and copy all png/jpg files recent modified? Or just all?
        # Copying specific folders
        cp -r "${reports_dir}/hidden_state_viz_${pooling_method}/"* "${images_dir}/" 2>/dev/null || true
        cp -r "${reports_dir}/critical_tokens_${pooling_method}/"* "${images_dir}/" 2>/dev/null || true
        
        echo "Copied images to ${images_dir}" >> "${exp_dir}/summary.txt"
    fi
}

# ---- Main Execution ----
echo "Experiment Runner"
echo "Base directory: ${EXPERIMENT_BASE_DIR}"

if [ "$DRY_RUN" = true ]; then
    mkdir -p "${EXPERIMENT_BASE_DIR}"
    echo "Dry run preview..." > "${EXPERIMENT_BASE_DIR}/README.txt"
else
    mkdir -p "${EXPERIMENT_BASE_DIR}"
fi

for i in "${!EXPERIMENTS[@]}"; do
    echo "[$((i+1))/${#EXPERIMENTS[@]}]"
    run_experiment "${EXPERIMENTS[$i]}"
done

echo "All experiments completed."
