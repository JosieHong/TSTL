#!/usr/bin/env python3
import numpy as np
import sys
import os
import random
from typing import Dict, Any, List


def load_npz_data(file_path: str) -> Dict[str, Any]:
    """Load molecular graph data from NPZ file."""
    try:
        with np.load(file_path, allow_pickle=True) as data:
            # Extract the data dictionary from the NPZ file
            molecular_data = data['data'][0]
        
        # DEBUG: Print shapes and types of loaded data
        for key in molecular_data.keys():
            print(key, molecular_data[key].shape if isinstance(molecular_data[key], np.ndarray) else type(molecular_data[key]))

        print(f"Loaded molecular data from {file_path}")
        print(f"Number of molecules: {len(molecular_data['label'])}")
        print(f"Available keys: {list(molecular_data.keys())}")
        
        return molecular_data
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        sys.exit(1)


def get_molecule_indices(molecular_data: Dict[str, Any]) -> List[int]:
    """Get indices for individual molecules from the molecular graph data."""
    n_nodes = molecular_data['n_node']
    n_edges = molecular_data['n_edge']
    
    # Calculate cumulative indices for nodes and edges
    node_cumsum = np.cumsum([0] + n_nodes.tolist())
    edge_cumsum = np.cumsum([0] + n_edges.tolist())
    
    molecule_indices = []
    for i in range(len(n_nodes)):
        molecule_indices.append({
            'mol_idx': i,
            'node_start': node_cumsum[i],
            'node_end': node_cumsum[i + 1],
            'edge_start': edge_cumsum[i],
            'edge_end': edge_cumsum[i + 1]
        })
    
    return molecule_indices


def sample_molecular_data(molecular_data: Dict[str, Any], sample_size: int, seed: int = 42) -> Dict[str, Any]:
    """Sample molecular graph data maintaining the structure."""
    n_molecules = len(molecular_data['label'])
    
    if sample_size > n_molecules:
        print(f"Warning: Sample size {sample_size} is larger than available molecules {n_molecules}")
        return molecular_data.copy()
    
    # Set seed for reproducible sampling
    random.seed(seed)
    np.random.seed(seed)
    
    # Get molecule indices
    molecule_indices = get_molecule_indices(molecular_data)
    
    # Randomly sample molecule indices
    sampled_mol_indices = sorted(random.sample(range(n_molecules), sample_size))
    
    # Initialize sampled data structure
    sampled_data = {
        'n_node': [],
        'n_edge': [],
        'node_attr': [],
        'edge_attr': [],
        'src': [],
        'dst': [],
        'label': []
    }
    
    for mol_idx in sampled_mol_indices:
        mol_info = molecule_indices[mol_idx]
        
        # Extract molecule-specific data
        n_node = molecular_data['n_node'][mol_idx]
        n_edge = molecular_data['n_edge'][mol_idx]
        label = molecular_data['label'][mol_idx]
        
        sampled_data['n_node'].append(n_node)
        sampled_data['n_edge'].append(n_edge)
        sampled_data['label'].append(label)
        
        # Extract node attributes
        node_attrs = molecular_data['node_attr'][mol_info['node_start']:mol_info['node_end']]
        sampled_data['node_attr'].append(node_attrs)
        
        # Extract edge attributes and connectivity
        edge_attrs = molecular_data['edge_attr'][mol_info['edge_start']:mol_info['edge_end']]
        src = molecular_data['src'][mol_info['edge_start']:mol_info['edge_end']]
        dst = molecular_data['dst'][mol_info['edge_start']:mol_info['edge_end']]
        
        sampled_data['edge_attr'].append(edge_attrs)
        sampled_data['src'].append(src)
        sampled_data['dst'].append(dst)
    
    # Convert lists back to the original format
    final_sampled_data = {}
    
    # Handle scalar arrays
    for key in ['n_node', 'n_edge']:
        final_sampled_data[key] = np.array(sampled_data[key]).astype(int)
    # Handle concatenated arrays
    final_sampled_data['node_attr'] = np.vstack(sampled_data['node_attr']).astype(bool)
    final_sampled_data['edge_attr'] = np.vstack(sampled_data['edge_attr']).astype(bool)
    final_sampled_data['src'] = np.hstack(sampled_data['src']).astype(int)
    final_sampled_data['dst'] = np.hstack(sampled_data['dst']).astype(int)
    # Handle labels
    final_sampled_data['label'] = np.array(sampled_data['label']).astype(float)
    
    # DEBUG: Print shapes and types of final sampled data
    for key in final_sampled_data.keys():
        print(key, final_sampled_data[key].shape if isinstance(final_sampled_data[key], np.ndarray) else type(final_sampled_data[key]))

    return final_sampled_data


def save_sampled_npz(sampled_data: Dict[str, Any], output_path: str) -> None:
    """Save sampled molecular data to NPZ file."""
    try:
        np.savez_compressed(output_path, data=[sampled_data])
        print(f"Saved {len(sampled_data['label'])} molecules to {output_path}")
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")


def print_data_summary(data: Dict[str, Any], title: str) -> None:
    """Print summary of molecular data."""
    print(f"\n{title}:")
    print(f"  Number of molecules: {len(data['label'])}")
    print(f"  Average nodes per molecule: {np.mean(data['n_node']):.2f}")
    print(f"  Average edges per molecule: {np.mean(data['n_edge']):.2f}")
    if len(data['label']) > 0:
        print(f"  Label range: {np.min(data['label']):.2f} - {np.max(data['label']):.2f}")


def main(): 
    # Parse arguments
    input_file = sys.argv[1]
    base_name = input_file.split('/')[-1].split('_')[0]
    output_dir = sys.argv[2]

    # Check command line arguments
    if len(sys.argv) < 4:
        print("Usage: python sample_npz_data.py <input.npz> <sample_size1> <sample_size2> ...")
        print("Example: python sample_npz_data.py dataset_graph_SMRT.npz 40 60 80 100 120 140 160 180 200")
        sys.exit(1)
    
    try:
        sample_sizes = [int(arg) for arg in sys.argv[3:]]
    except ValueError:
        print("Error: All sample sizes must be integers")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    # Load the original molecular data
    original_data = load_npz_data(input_file)
    
    # Print summary of original data
    print_data_summary(original_data, "Original Data Summary")
    
    print(f"\nProcessing {len(sample_sizes)} different sample sizes...")
    
    # Process each sample size
    for sample_size in sample_sizes:
        print(f"\nProcessing sample size: {sample_size}")
        
        # Sample the molecular data
        sampled_data = sample_molecular_data(original_data, sample_size)
        
        # Print summary of sampled data
        print_data_summary(sampled_data, f"Sampled Data (n={sample_size})")
        
        # Create output filename
        output_file = f"{output_dir}/{base_name}_graph_sample{sample_size}.npz"
        
        # Save the sampled data
        save_sampled_npz(sampled_data, output_file)
    
    print(f"\nCompleted! Generated {len(sample_sizes)} sampled datasets.")
    print("Output files:")
    for size in sample_sizes:
        print(f"  - {output_dir}/{base_name}_graph_sample{size}.npz")

if __name__ == "__main__":
    main()