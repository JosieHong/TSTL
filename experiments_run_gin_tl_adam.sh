#!/bin/bash

# Directory containing the dataset files
DATA_DIR="./data/processed_graph"

# Number of GPUs to use
NUM_GPUS=4

# Check if the directory exists
if [ ! -d "$DATA_DIR" ]; then
  echo "Error: Directory $DATA_DIR does not exist."
  exit 1
fi

# Extract all subset names from the filenames
echo "Extracting subset names from $DATA_DIR..."
SUBSETS=()
for file in "$DATA_DIR"/dataset_graph_*.npz; do
  if [ -f "$file" ]; then
    # Extract the report ID from the filename (e.g., report0009)
    subset=$(basename "$file" | sed 's/dataset_graph_\(.*\)\.npz/\1/')
    
    # Skip SMRT if it's found as a subset
    if [ "$subset" != "SMRT" ]; then
      SUBSETS+=("$subset")
    fi
  fi
done

# Check if any subsets were found
if [ ${#SUBSETS[@]} -eq 0 ]; then
  echo "Error: No subsets found in $DATA_DIR."
  exit 1
fi

# Sort the subsets for more organized execution
IFS=$'\n' SORTED_SUBSETS=($(sort <<<"${SUBSETS[*]}"))
unset IFS

# Total number of experiments to run
TOTAL=${#SORTED_SUBSETS[@]}
echo "Found $TOTAL subsets to process."

# Function to run experiments on a specific GPU
run_on_gpu() {
  local gpu_id=$1
  local start_idx=$2
  local end_idx=$3
  local subsets=("${@:4}")
  
  for i in $(seq $start_idx $end_idx); do
    if [ $i -ge ${#subsets[@]} ]; then
      break
    fi
    
    subset="${subsets[$i]}"
    echo "[GPU $gpu_id - $((i-start_idx+1))/$(($end_idx-$start_idx+1))] Running experiment on subset: $subset"
    
    # Run the experiment on the specified GPU using the --device parameter
    python run_tl.py -c SMRT -t "$subset" -s 42 --n_folds 5 --device $gpu_id --opt adam --results_dir "./results_run_all" 
    
    # Check if the command completed successfully
    if [ $? -eq 0 ]; then
      echo "[GPU $gpu_id - SUCCESS] Experiment on $subset completed successfully."
    else
      echo "[GPU $gpu_id - ERROR] Failed running experiment on $subset. Continuing with next subset..."
    fi
    
    echo "-----------------------------------------"
  done
}

# Calculate number of subsets per GPU
echo "Dividing $TOTAL subsets across $NUM_GPUS GPUs..."
subsets_per_gpu=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

# Launch processes for each GPU in parallel
for gpu_id in $(seq 0 $(($NUM_GPUS-1))); do
  start_idx=$(( gpu_id * subsets_per_gpu ))
  end_idx=$(( start_idx + subsets_per_gpu - 1 ))
  
  # Make sure end_idx doesn't exceed array bounds
  if [ $end_idx -ge $TOTAL ]; then
    end_idx=$(( TOTAL - 1 ))
  fi
  
  echo "Starting GPU $gpu_id with subsets $start_idx to $end_idx"
  
  # Run in background with &
  run_on_gpu $gpu_id $start_idx $end_idx "${SORTED_SUBSETS[@]}" &
  
  # Store the process ID
  gpu_pids[$gpu_id]=$!
done

# Wait for all GPU processes to complete
echo "All GPU processes launched. Waiting for completion..."
for pid in ${gpu_pids[@]}; do
  wait $pid
done

echo "All experiments completed!"