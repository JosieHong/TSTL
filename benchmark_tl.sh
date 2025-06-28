#!/bin/bash

# Default configuration
DEFAULT_DATASETS=("SMRT" "report0390" "report0391" "report0392" "report0262" "report0383" "report0382" "report0263")
MODELS=("MPNN" "GAT" "GTN" "GIN" "GIN3" "3DMol")
GPU_ASSIGNMENTS=(4 4 5 5 6 6) # GPU assignments for each model

# Function to display usage
usage() {
    echo "Usage: $0 <target_dataset>"
    echo "  target_dataset: Dataset for retention time prediction (e.g., report0063)"
    echo "  The script will run 5-fold cross-validation experiments on the specified dataset" 
    echo "  from the pre-trained models across all source datasets (report0186, report0390, report0391)"
    exit 1
}

# Process command line arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ $# -eq 0 ]; then
    usage
fi

TARGET_DATASET="$1"
DATASETS=("${DEFAULT_DATASETS[@]}")

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to run a single experiment
run_experiment() {
    local model=$1
    local device=$2
    local source_dataset=$3
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    echo "Starting $model on GPU $device with source dataset $source_dataset and target dataset $TARGET_DATASET..."
    # Change this line to store PID directly
    (python -u run_tl.py \
        -t "$TARGET_DATASET" \
        -c "$source_dataset" \
        --gnn "$model" \
        --opt adam \
        --device "$device" \
        --n_folds 5 > "./logs/${model}_tl_${source_dataset}_${TARGET_DATASET}_${timestamp}.log" 2>&1) &
    
    # Store the process ID of the most recently started background process
    echo $! > "${model}_${TARGET_DATASET}_tl.pid"
}

# Function to wait for current batch to complete
wait_for_batch() {
    echo "Waiting for all processes to complete..."
    wait
    rm -f *_${TARGET_DATASET}_tl.pid
}

# Main execution loop
echo "Running transfer learning experiments with target dataset: $TARGET_DATASET"
for source_dataset in "${DATASETS[@]}"; do
    echo "Source dataset: $source_dataset"
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Start each model on its designated GPU
    for i in "${!MODELS[@]}"; do
        run_experiment "${MODELS[i]}" "${GPU_ASSIGNMENTS[i]}" "$source_dataset"
    done
    
    # Wait for all processes in current batch to complete
    wait_for_batch
done

echo "All experiments have completed!"
