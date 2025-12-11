#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Experiment Runner Script
# ============================================================================
# This script runs multiple experiments with different parameter combinations
# and stores logs for comparison.
#
# Usage:
#   ./run_experiments.sh
#   ./run_experiments.sh --dry-run  # Preview experiments without running
#
# Customization:
#   Edit the EXPERIMENTS array below to add/remove/modify parameter combinations.
#   Each entry is a space-separated list of PARAM=value assignments.
#
#   Available parameters to vary:
#     - POOLING_METHOD: mean, last_token, per_token
#     - TOKEN_SELECTION_METHOD: last_token, gradient, dp_gradient, dp_average, token_mlp
#     - LAYER_SELECTION_METHOD: fixed, mlp
#     - LAYERS: space-separated list of layer numbers (e.g., "26 27 28")
#     - MAX_SAMPLES: number of samples to process
#     - SKIP_* flags added in run.bash: SKIP_HIDDEN, SKIP_PROBES, SKIP_PLOTS,
#       SKIP_CRITICAL, SKIP_STEERING, SKIP_EVAL, SKIP_EMBED (set to 1 to skip)
#     - Note: SKIP_EMBED is always forced to 0 (Step 8 always runs)
#     - And any other parameters defined in run.bash
#
#   DP Alignment (--dp-alignment) is automatically configured:
#     - Enabled if TOKEN_SELECTION_METHOD is dp_gradient or dp_average (required)
#     - Enabled if POOLING_METHOD is per_token (useful for better alignments)
#     - Disabled otherwise (not needed for last_token/gradient or mean/last_token pooling)
#
#   After each experiment, all generated images are automatically copied to:
#     experiments/<timestamp>/<experiment_name>/images/
# ============================================================================

# ---- Configuration ----

# Base directory for experiment logs
EXPERIMENTS_DIR="experiments"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_BASE_DIR="${EXPERIMENTS_DIR}/${TIMESTAMP}"

# Source the original run.bash to get default values and setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_BASH="${SCRIPT_DIR}/run.bash"

# Check if run.bash exists
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
# Define parameter combinations to test
# Each array element is a space-separated list of parameter assignments
# Format: "PARAM1=value1 PARAM2=value2 ..."

declare -a EXPERIMENTS=(

    # Test different token selection methods with per_token pooling
    "POOLING_METHOD=per_token TOKEN_SELECTION_METHOD=last_token LAYER_SELECTION_METHOD=fixed SKIP_HIDDEN=0 SKIP_PROBES=0 SKIP_PLOTS=0 SKIP_CRITICAL=0 SKIP_STEERING=0 SKIP_EVAL=0"
    "POOLING_METHOD=per_token TOKEN_SELECTION_METHOD=gradient LAYER_SELECTION_METHOD=fixed SKIP_HIDDEN=1 SKIP_PROBES=1 SKIP_PLOTS=0 SKIP_CRITICAL=1 SKIP_STEERING=0 SKIP_EVAL=0"
    "POOLING_METHOD=per_token TOKEN_SELECTION_METHOD=dp_gradient LAYER_SELECTION_METHOD=fixed SKIP_HIDDEN=1 SKIP_PROBES=1 SKIP_PLOTS=0 SKIP_CRITICAL=1 SKIP_STEERING=0 SKIP_EVAL=0"
    "POOLING_METHOD=per_token TOKEN_SELECTION_METHOD=dp_average LAYER_SELECTION_METHOD=fixed SKIP_HIDDEN=1 SKIP_PROBES=1 SKIP_PLOTS=0 SKIP_CRITICAL=1 SKIP_STEERING=0 SKIP_EVAL=0"
    
    # Test different pooling methods with mean & last_token selection
    "POOLING_METHOD=last_token TOKEN_SELECTION_METHOD=last_token LAYER_SELECTION_METHOD=fixed SKIP_HIDDEN=0 SKIP_PROBES=0 SKIP_PLOTS=0 SKIP_CRITICAL=0 SKIP_STEERING=0 SKIP_EVAL=0"    
    "POOLING_METHOD=mean TOKEN_SELECTION_METHOD=last_token LAYER_SELECTION_METHOD=fixed SKIP_HIDDEN=0 SKIP_PROBES=0 SKIP_PLOTS=0 SKIP_CRITICAL=0 SKIP_STEERING=0 SKIP_EVAL=0"
    "POOLING_METHOD=mean TOKEN_SELECTION_METHOD=gradient LAYER_SELECTION_METHOD=fixed SKIP_HIDDEN=1 SKIP_PROBES=1 SKIP_PLOTS=0 SKIP_CRITICAL=0 SKIP_STEERING=0 SKIP_EVAL=0"
    "POOLING_METHOD=mean TOKEN_SELECTION_METHOD=dp_gradient LAYER_SELECTION_METHOD=fixed SKIP_HIDDEN=1 SKIP_PROBES=1 SKIP_PLOTS=0 SKIP_CRITICAL=0 SKIP_STEERING=0 SKIP_EVAL=0"
    
    
    # Test with token_mlp if model exists (uncomment if you have the model)
    # "POOLING_METHOD=per_token TOKEN_SELECTION_METHOD=token_mlp LAYER_SELECTION_METHOD=fixed SKIP_HIDDEN=1 SKIP_PROBES=1 SKIP_PLOTS=1 SKIP_CRITICAL=1 SKIP_STEERING=0 SKIP_EVAL=0"
    
    # Test layer selection methods (if you have layer selector model)
    # "POOLING_METHOD=per_token TOKEN_SELECTION_METHOD=last_token LAYER_SELECTION_METHOD=mlp SKIP_HIDDEN=1 SKIP_PROBES=1 SKIP_PLOTS=1 SKIP_CRITICAL=1 SKIP_STEERING=0 SKIP_EVAL=0"
)

# ---- Helper Functions ----

# Generate experiment name from parameters
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

# Create a modified copy of run.bash with parameter overrides
create_modified_run_script() {
    local exp_params="$1"
    local output_script="$2"
    
    # Copy the original run.bash and ensure Unix line endings
    # Use dos2unix if available, otherwise use sed/tr to convert
    if command -v dos2unix >/dev/null 2>&1; then
        cp "${RUN_BASH}" "${output_script}"
        dos2unix "${output_script}" >/dev/null 2>&1 || true
    else
        # Convert line endings manually using tr or sed
        tr -d '\r' < "${RUN_BASH}" > "${output_script}" 2>/dev/null || \
        sed 's/\r$//' "${RUN_BASH}" > "${output_script}" 2>/dev/null || \
        cp "${RUN_BASH}" "${output_script}"
    fi
    
    # Override each parameter by replacing the assignment line
    for param in $exp_params; do
        local key="${param%%=*}"
        local value="${param#*=}"
        
        # Escape special characters in the key for sed
        local escaped_key=$(printf '%s\n' "$key" | sed 's/[[\.*^$()+?{|]/\\&/g')
        
        # Find the line with the parameter and replace it
        # Match lines like: KEY="value" or KEY=value and replace with KEY="value"
        # Use a temporary file to avoid issues with sed -i on some systems
        local temp_file=$(mktemp)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS sed
            sed "s|^${escaped_key}=.*|${key}=\"${value}\"|" "${output_script}" > "${temp_file}"
        else
            # Linux sed
            sed "s|^${escaped_key}=.*|${key}=\"${value}\"|" "${output_script}" > "${temp_file}"
        fi
        mv "${temp_file}" "${output_script}"
    done
    
    # Ensure Unix line endings again after modifications
    if command -v dos2unix >/dev/null 2>&1; then
        dos2unix "${output_script}" >/dev/null 2>&1 || true
    else
        tr -d '\r' < "${output_script}" > "${temp_file}" 2>/dev/null && mv "${temp_file}" "${output_script}" || true
    fi
    
    # Verify the script starts with shebang
    if ! head -n 1 "${output_script}" | grep -q "^#!/"; then
        echo "Warning: Modified script may be missing shebang" >&2
    fi
    
    # Verify line 2 has the set command
    local line2=$(sed -n '2p' "${output_script}")
    if [[ ! "$line2" =~ ^set.*pipefail ]]; then
        echo "Warning: Line 2 of script may be incorrect: ${line2}" >&2
    fi
    
    # Force SKIP_EMBED=0 to ensure Step 8 always runs (override any user setting)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|^SKIP_EMBED=.*|SKIP_EMBED=0|" "${output_script}"
    else
        sed -i "s|^SKIP_EMBED=.*|SKIP_EMBED=0|" "${output_script}"
    fi
    
    # Conditionally set DP_ALIGNMENT_ARGS based on methods used
    # Extract TOKEN_SELECTION_METHOD and POOLING_METHOD from parameters
    local token_selection_method="last_token"
    local pooling_method="per_token"
    
    for param in $exp_params; do
        local key="${param%%=*}"
        local value="${param#*=}"
        if [ "$key" = "TOKEN_SELECTION_METHOD" ]; then
            token_selection_method="$value"
        elif [ "$key" = "POOLING_METHOD" ]; then
            pooling_method="$value"
        fi
    done
    
    # DP alignment is needed if:
    # 1. TOKEN_SELECTION_METHOD is dp_gradient or dp_average (required for Step 6)
    # 2. POOLING_METHOD is per_token (useful for Step 2, though not strictly required)
    local needs_dp_alignment=false
    if [[ "$token_selection_method" == "dp_gradient" ]] || [[ "$token_selection_method" == "dp_average" ]]; then
        needs_dp_alignment=true
    elif [[ "$pooling_method" == "per_token" ]]; then
        needs_dp_alignment=true
    fi
    
    # Set DP_ALIGNMENT_ARGS accordingly
    if [ "$needs_dp_alignment" = true ]; then
        # Set to use DP alignment
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|^DP_ALIGNMENT_ARGS=.*|DP_ALIGNMENT_ARGS=(--dp-alignment)|" "${output_script}"
        else
            sed -i "s|^DP_ALIGNMENT_ARGS=.*|DP_ALIGNMENT_ARGS=(--dp-alignment)|" "${output_script}"
        fi
    else
        # Remove DP alignment (set to empty array)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|^DP_ALIGNMENT_ARGS=.*|DP_ALIGNMENT_ARGS=()|" "${output_script}"
        else
            sed -i "s|^DP_ALIGNMENT_ARGS=.*|DP_ALIGNMENT_ARGS=()|" "${output_script}"
        fi
    fi
    
    # Make it executable
    chmod +x "${output_script}" 2>/dev/null || true
}

# Run a single experiment
run_experiment() {
    local exp_params="$1"
    local exp_name=$(generate_experiment_name "$exp_params")
    local exp_dir="${EXPERIMENT_BASE_DIR}/${exp_name}"
    local log_file="${exp_dir}/run.log"
    local summary_file="${exp_dir}/summary.txt"
    
    echo ""
    echo "=========================================="
    echo "Running experiment: ${exp_name}"
    echo "Parameters: ${exp_params}"
    echo "=========================================="
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would create: ${exp_dir}"
        echo "  [DRY RUN] Would log to: ${log_file}"
        return 0
    fi
    
    # Create experiment directory
    mkdir -p "${exp_dir}"
    
    # Create modified run script with parameter overrides
    local modified_script="${exp_dir}/run_modified.sh"
    create_modified_run_script "$exp_params" "$modified_script"
    
    # Record experiment metadata
    {
        echo "Experiment: ${exp_name}"
        echo "Timestamp: $(date)"
        echo "Parameters:"
        for param in $exp_params; do
            echo "  ${param}"
        done
        echo ""
        echo "Modified script: ${modified_script}"
        echo ""
    } > "${summary_file}"
    
    # Run the experiment and capture output
    local script_dir=$(dirname "${RUN_BASH}")
    local start_time=$(date +%s)
    
    # Verify script exists and is readable
    if [ ! -f "${modified_script}" ]; then
        echo "Error: Modified script not found at ${modified_script}" >&2
        return 1
    fi
    
    # Check script syntax before running
    if ! bash -n "${modified_script}" 2>&1; then
        echo "Error: Modified script has syntax errors" >&2
        echo "First 5 lines of script:" >&2
        head -n 5 "${modified_script}" >&2
        return 1
    fi
    
    # Run from script directory to ensure relative paths in run.bash work
    # Get absolute path to the modified script
    local exp_dir_abs=$(cd "$(dirname "${modified_script}")" && pwd)
    local script_name=$(basename "${modified_script}")
    local script_to_run="${exp_dir_abs}/${script_name}"
    
    # Verify the absolute path exists
    if [ ! -f "${script_to_run}" ]; then
        echo "Error: Cannot resolve script path: ${script_to_run}" >&2
        return 1
    fi
    
    # Run bash explicitly with the absolute path, from the original script directory
    # This ensures relative paths in run.bash work correctly
    if (cd "${script_dir}" && /bin/bash "${script_to_run}") 2>&1 | tee "${log_file}"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "Experiment completed successfully in ${duration} seconds" >> "${summary_file}"
        echo "Status: SUCCESS" >> "${summary_file}"
        echo "✓ Experiment '${exp_name}' completed successfully"
        
        # Copy all generated images to experiment directory
        copy_experiment_images "$exp_params" "$exp_dir" "$script_dir"
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "Experiment failed after ${duration} seconds" >> "${summary_file}"
        echo "Status: FAILED" >> "${summary_file}"
        echo "✗ Experiment '${exp_name}' failed (check ${log_file})"
        # Don't exit on failure, continue with other experiments
        # Still try to copy images even if experiment failed (might have partial results)
        copy_experiment_images "$exp_params" "$exp_dir" "$script_dir"
    fi
    
    # Keep the modified script for reference (comment out if you want to clean it up)
    # rm -f "${modified_script}"
}

# Copy all generated images to experiment directory
copy_experiment_images() {
    local exp_params="$1"
    local exp_dir="$2"
    local script_dir="$3"
    
    # Extract POOLING_METHOD from parameters or use default
    local pooling_method="per_token"
    for param in $exp_params; do
        local key="${param%%=*}"
        local value="${param#*=}"
        if [ "$key" = "POOLING_METHOD" ]; then
            pooling_method="$value"
            break
        fi
    done
    
    # Get HF_HOME from the modified script (it's set in run.bash)
    # Default to the common path if we can't extract it
    local hf_home="/nobackup/bdeka/huggingface_cache"
    if [ -f "${script_dir}/run.bash" ]; then
        local extracted_hf=$(grep "^export HF_HOME=" "${script_dir}/run.bash" | head -1 | sed 's/.*="\([^"]*\)".*/\1/' || echo "")
        if [ -n "$extracted_hf" ]; then
            hf_home="$extracted_hf"
        fi
    fi
    
    # Create images directory in experiment folder
    local images_dir="${exp_dir}/images"
    mkdir -p "${images_dir}"
    
    echo "  Copying generated images to ${images_dir}..."
    local images_copied=0
    
    # Copy Step 4 images: probe visualizations
    local plot_output="${script_dir}/reports/hidden_state_viz_${pooling_method}"
    if [ -d "${plot_output}" ]; then
        find "${plot_output}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.svg" \) -exec cp {} "${images_dir}/" \; 2>/dev/null
        local count=$(find "${plot_output}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.svg" \) 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            images_copied=$((images_copied + count))
            echo "    Copied ${count} image(s) from Step 4 (probe visualizations)"
        fi
    fi
    
    # Copy Step 5 images: critical token analysis
    local critical_output="${script_dir}/reports/critical_tokens_${pooling_method}"
    if [ -d "${critical_output}" ]; then
        find "${critical_output}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.svg" \) -exec cp {} "${images_dir}/" \; 2>/dev/null
        local count=$(find "${critical_output}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.svg" \) 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            images_copied=$((images_copied + count))
            echo "    Copied ${count} image(s) from Step 5 (critical tokens)"
        fi
    fi
    
    # Copy Step 8 images: embedding comparisons (in-sample and out-of-sample)
    local embed_in="${hf_home}/reports/steering_embedding_comparison_in_sample"
    local embed_out="${hf_home}/reports/steering_embedding_comparison_out_of_sample"
    
    if [ -d "${embed_in}" ]; then
        find "${embed_in}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.svg" \) -exec cp {} "${images_dir}/" \; 2>/dev/null
        local count=$(find "${embed_in}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.svg" \) 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            images_copied=$((images_copied + count))
            echo "    Copied ${count} image(s) from Step 8 (embedding comparison in-sample)"
        fi
    fi
    
    if [ -d "${embed_out}" ]; then
        find "${embed_out}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.svg" \) -exec cp {} "${images_dir}/" \; 2>/dev/null
        local count=$(find "${embed_out}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.svg" \) 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            images_copied=$((images_copied + count))
            echo "    Copied ${count} image(s) from Step 8 (embedding comparison out-of-sample)"
        fi
    fi
    
    if [ "$images_copied" -gt 0 ]; then
        echo "  ✓ Copied ${images_copied} total image(s) to ${images_dir}/"
        echo "Images copied: ${images_copied}" >> "${exp_dir}/summary.txt"
    else
        echo "  ⚠ No images found to copy"
    fi
}

# ---- Main Execution ----

echo "=========================================="
echo "Experiment Runner"
echo "=========================================="
echo "Base directory: ${EXPERIMENT_BASE_DIR}"
echo "Number of experiments: ${#EXPERIMENTS[@]}"
echo ""

if [ "$DRY_RUN" = true ]; then
    # In dry-run mode, create a preview directory structure
    mkdir -p "${EXPERIMENT_BASE_DIR}"
    
    # Create preview README
    {
        echo "Experiment Run Summary (DRY RUN PREVIEW)"
        echo "========================================"
        echo "Timestamp: ${TIMESTAMP}"
        echo "Total experiments: ${#EXPERIMENTS[@]}"
        echo ""
        echo "This is a preview. No experiments were actually run."
        echo ""
        echo "Experiments that would be run:"
        for i in "${!EXPERIMENTS[@]}"; do
            exp_name=$(generate_experiment_name "${EXPERIMENTS[$i]}")
            echo "  $((i+1)). ${exp_name}"
            echo "     ${EXPERIMENTS[$i]}"
        done
        echo ""
    } > "${EXPERIMENT_BASE_DIR}/README.txt"
    
    # Create preview directories for each experiment
    for i in "${!EXPERIMENTS[@]}"; do
        exp_name=$(generate_experiment_name "${EXPERIMENTS[$i]}")
        exp_dir="${EXPERIMENT_BASE_DIR}/${exp_name}"
        mkdir -p "${exp_dir}"
        echo "[DRY RUN] Created preview directory: ${exp_dir}" > "${exp_dir}/PREVIEW.txt"
        echo "This directory would contain:" >> "${exp_dir}/PREVIEW.txt"
        echo "  - run.log (full execution log)" >> "${exp_dir}/PREVIEW.txt"
        echo "  - summary.txt (experiment metadata)" >> "${exp_dir}/PREVIEW.txt"
        echo "  - run_modified.sh (modified run.bash script)" >> "${exp_dir}/PREVIEW.txt"
    done
    
    echo ""
    echo "Preview directory structure created at: ${EXPERIMENT_BASE_DIR}"
    echo "Remove it with: rm -rf ${EXPERIMENT_BASE_DIR}"
else
    mkdir -p "${EXPERIMENT_BASE_DIR}"
    
    # Create master summary file
    {
        echo "Experiment Run Summary"
        echo "======================"
        echo "Timestamp: ${TIMESTAMP}"
        echo "Total experiments: ${#EXPERIMENTS[@]}"
        echo ""
        echo "Experiments:"
        for i in "${!EXPERIMENTS[@]}"; do
            exp_name=$(generate_experiment_name "${EXPERIMENTS[$i]}")
            echo "  $((i+1)). ${exp_name}"
            echo "     ${EXPERIMENTS[$i]}"
        done
        echo ""
    } > "${EXPERIMENT_BASE_DIR}/README.txt"
fi

# Run each experiment
for i in "${!EXPERIMENTS[@]}"; do
    echo ""
    echo "[$((i+1))/${#EXPERIMENTS[@]}]"
    run_experiment "${EXPERIMENTS[$i]}"
done

# Generate comparison summary
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "=========================================="
    echo "Generating preview comparison summary..."
    echo "=========================================="
    
    comparison_file="${EXPERIMENT_BASE_DIR}/comparison.txt"
    {
        echo "Experiment Comparison Summary (DRY RUN PREVIEW)"
        echo "================================================"
        echo "Generated: $(date)"
        echo ""
        echo "This is a preview. No experiments were actually run."
        echo ""
        echo "Experiments that would be compared:"
        for i in "${!EXPERIMENTS[@]}"; do
            exp_name=$(generate_experiment_name "${EXPERIMENTS[$i]}")
            echo "  $((i+1)). ${exp_name}"
            echo "     ${EXPERIMENTS[$i]}"
        done
        echo ""
        echo "After running experiments, this file would contain:"
        echo "  - Status of each experiment (SUCCESS/FAILED)"
        echo "  - Execution time for each experiment"
        echo "  - Key metrics extracted from logs"
        echo "  - Comparison of results across experiments"
    } > "${comparison_file}"
    
    echo "Preview comparison summary saved to: ${comparison_file}"
elif [ "$DRY_RUN" = false ]; then
    echo ""
    echo "=========================================="
    echo "Generating comparison summary..."
    echo "=========================================="
    
    comparison_file="${EXPERIMENT_BASE_DIR}/comparison.txt"
    {
        echo "Experiment Comparison Summary"
        echo "============================="
        echo "Generated: $(date)"
        echo ""
        
        for exp_dir in "${EXPERIMENT_BASE_DIR}"/*/; do
            if [ -d "$exp_dir" ]; then
                exp_name=$(basename "$exp_dir")
                echo "Experiment: ${exp_name}"
                echo "----------------------------------------"
                
                if [ -f "${exp_dir}/summary.txt" ]; then
                    cat "${exp_dir}/summary.txt"
                fi
                
                # Extract key metrics from log if available
                if [ -f "${exp_dir}/run.log" ]; then
                    echo ""
                    echo "Key outputs from log:"
                    # Look for common success indicators
                    if grep -q "Done!" "${exp_dir}/run.log"; then
                        echo "  ✓ Pipeline completed"
                    fi
                    if grep -q "Steering evaluation" "${exp_dir}/run.log"; then
                        echo "  ✓ Steering evaluation completed"
                    fi
                    if grep -q "Error\|ERROR\|Failed\|FAILED" "${exp_dir}/run.log"; then
                        echo "  ✗ Errors detected in log"
                    fi
                fi
                echo ""
                echo ""
            fi
        done
    } > "${comparison_file}"
    
    echo "Comparison summary saved to: ${comparison_file}"
fi

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results directory: ${EXPERIMENT_BASE_DIR}"
echo ""
echo "To compare results:"
echo "  cat ${EXPERIMENT_BASE_DIR}/comparison.txt"
echo ""
echo "To view a specific experiment log:"
echo "  cat ${EXPERIMENT_BASE_DIR}/<experiment_name>/run.log"
echo ""

