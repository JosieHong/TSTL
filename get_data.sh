#!/bin/bash

# Create directories
mkdir -p data
cd data

# Clone RepoRT repository
git clone https://github.com/michaelwitting/RepoRT.git

# Download SMRT dataset
wget https://figshare.com/ndownloader/files/18130625 -O SMRT_dataset.sdf

# Run preprocessing script
cd ..
python preprocess_graph.py
python preprocess_geom.py

# Plot data distribution
python plot_data_dist.py