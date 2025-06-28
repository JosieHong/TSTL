#!/usr/bin/env python3
import pickle
import random
import sys
import os
from typing import List, Any


def load_pickle_data(file_path: str) -> List[Any]:
    """Load data from pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Pickle file must contain a list")
        
        print(f"Loaded {len(data)} items from {file_path}")
        return data
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)


def sample_data(data: List[Any], sample_size: int, seed: int = 42) -> List[Any]:
    """Sample data with given size."""
    if sample_size > len(data):
        print(f"Warning: Sample size {sample_size} is larger than data size {len(data)}")
        return data.copy()
    
    # Set seed for reproducible sampling
    random.seed(seed)
    return random.sample(data, sample_size)


def save_sampled_data(sampled_data: List[Any], output_path: str) -> None:
    """Save sampled data to pickle file."""
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(sampled_data, f)
        print(f"Saved {len(sampled_data)} items to {output_path}")
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")

def main():
    # Default input file name (you can modify this)
    input_file = sys.argv[1]
    base_name = input_file.split('/')[-1].split('_')[0]
    output_dir = sys.argv[2]

    # Check command line arguments
    if len(sys.argv) < 4:
        print("Usage: python preprocess_limited_sizes.py <sample_size1> <sample_size2> ...")
        print("Example: python preprocess_limited_sizes.py 40 60 80 100 120 140 160 180 200")
        sys.exit(1)
    
    # Parse sample sizes from command line arguments
    try:
        sample_sizes = [int(arg) for arg in sys.argv[3:]]
    except ValueError:
        print("Error: All arguments must be integers")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    # Load the original data
    original_data = load_pickle_data(input_file)
    
    print(f"Processing {len(sample_sizes)} different sample sizes...")
    
    # Process each sample size
    for sample_size in sample_sizes:
        print(f"\nProcessing sample size: {sample_size}")
        
        # Sample the data
        sampled_data = sample_data(original_data, sample_size)
        
        # Create output filename
        output_file = f"{output_dir}/{base_name}_geom_sample{sample_size}.pkl"
        
        # Save the sampled data
        save_sampled_data(sampled_data, output_file)
    
    print(f"\nCompleted! Generated {len(sample_sizes)} sampled datasets.")
    print("Output files:")
    for size in sample_sizes:
        print(f"  - {output_dir}/{base_name}_geom_sample{size}.pkl")


if __name__ == "__main__":
    main()