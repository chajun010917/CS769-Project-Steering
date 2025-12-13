# Gating Steering Project

A comprehensive framework for analyzing and steering Large Language Models (LLMs) towards better reasoning capabilities using activation steering.

## Overview
This repository contains the codebase for analyzing critical tokens, computing probes, and evaluating steering vectors to improve model reasoning. The project uses the Llama-3.1-8B-Instruct model and evaluates on the MMLU-Pro-CoT dataset.

## Setup

1.  **Environment Setup**:
    Run the provided setup script to create the conda environment and install dependencies:
    ```bash
    bash setup.sh
    conda activate steering
    ```

2.  **Dataset Preparation**:
    Ensure the manually reviewed triples file is present at `artifacts/manual_review/12062025_human_review.json`.

## Usage

### Running standard experiments
The main entry point for running the experiment pipeline is `run.bash`. It handles dataset preparation, hidden state capture, probe computation, and steering evaluation.

To run the full pipeline:
```bash
./run.bash
```

#### Configuration
You can customize the execution by setting environment variables either before the command or by editing `run.bash`.

**Key Variables:**
-   `POOLING_METHOD`: `per_token` (default), `mean`, or `last_token`.
-   `TOKEN_SELECTION_METHOD`: `last_token` (default), `gradient`.
-   `MAX_SAMPLES`: Number of samples to process (default: 100).
-   `LAYERS`: Space-separated list of layers to analyze (default: "26 27 28 29 30 31").

**Skip Flags (set to 1 to skip):**
-   `SKIP_HIDDEN`: Skip Step 2 (Hidden State Capture).
-   `SKIP_PROBES`: Skip Step 3 (Probe Computation).
-   `SKIP_PLOTS`: Skip Step 4 (Visualization).
-   `SKIP_CRITICAL`: Skip Step 5 (Critical Token Analysis).
-   `SKIP_STEERING`: Skip Step 6 (Steering Vector Computation).
-   `SKIP_EVAL`: Skip Step 7 (Steering Evaluation).

Example:
```bash
# Run with 'mean' pooling and skip hidden state collection
export POOLING_METHOD="per_token"
export TOKEN_SELECTION_METHOD="last_token"
export SKIP_HIDDEN=1
./run.bash
```

To run a suite of experiments with different parameters:
```bash
./run_experiments.sh
```

#### Currently Configured Experiments
The `run_experiments.sh` script is pre-configured with the following experiments. You can add more by editing the `EXPERIMENTS` array in the script.

1.  **Baseline**:
    *   `POOLING_METHOD`: `per_token`
    *   `TOKEN_SELECTION_METHOD`: `last_token`
    *   Everything else standard (all steps enabled).

### Running Evaluation Suite
For detailed evaluation including alpha sweeps and sample size comparisons, use `run_eval.bash`:
```bash
./run_eval.bash
```

#### Evaluation Defaults
The `run_eval.bash` script uses the following default configuration (Baseline):
*   **Layer**: 28
*   **Token Selection**: `last_token`
*   **Samples**: 100
*   **Alpha Sweep**: 0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0

You can override these by setting environment variables, e.g.:
```bash
export LAYERS="29"
export TOKEN_SELECTION_METHOD="gradient"
./run_eval.bash
```

## Reproducibility
The code is well-organized, including all experiment scripts (e.g., bash scripts, python files). All experimental results can be easily reproducible without errors. 

## Contribution Statement:
Each team member worked on an individual branch, which were later merged into the main branch. caclassen is one of the authors of Zeng et al. 2025 (the dataset we use) and contributor made from xinyu-jiao is a commit made by Zhizhe Liu from his friend's laptop.