#!/bin/bash

# Model and GPU configurations
MODELS=("GIN" "MPNN" "GAT" "GTN" "GIN3" "3DMol")
GPU_ASSIGNMENTS=(4 4 5 5 6 6) # GPU assignments for each model

# Function to display usage
usage() {
    echo "Usage: $0 <target_dataset>"
    echo "  target_dataset: Dataset for retention time prediction (e.g., report0063)"
    echo "  The script will run 5-fold cross-validation experiments on the specified dataset"
    echo "  using task-specific transfer learning across all source datasets (SMRT, report0390, report0391)"
    exit 1
}

# Process command line arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ $# -eq 0 ]; then
    usage
fi

TARGET_DATASET="$1"

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to run a single experiment
run_experiment() {
    local model=$1
    local device=$2
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    echo "Starting $model on GPU $device for dataset $TARGET_DATASET..."
    python -u run_tstl.py \
        -c SMRT report0390 report0391 report0392 report0262 report0383 report0382 report0263 \
        -t "$TARGET_DATASET" \
        --gnn "$model" \
        --opt adam \
        --device "$device" \
        --n_folds 5 > "./logs/${model}_tstl_${TARGET_DATASET}_${timestamp}.log" 2>&1 &
    
    # Store the process ID
    echo $! > "${model}_${TARGET_DATASET}_tstl.pid"
}

# Function to wait for all processes to complete
wait_for_batch() {
    echo "Waiting for all processes to complete..."
    wait
    rm -f *_${TARGET_DATASET}_tstl.pid
}

# Main execution
echo "Running task specific transfer learning experiments with target dataset: $TARGET_DATASET"
timestamp=$(date +%Y%m%d_%H%M%S)

# Start each model on its designated GPU
for i in "${!MODELS[@]}"; do
    run_experiment "${MODELS[i]}" "${GPU_ASSIGNMENTS[i]}"
done

# Wait for all processes to complete
wait_for_batch

echo "All experiments have completed!"
