#!/bin/bash

# Sample size experiments with GIN + adam
# Sample sizes: 40, 60, 80, 100, 120 

# ==============================
# Sample data preparation
# ==============================
python sample_pkl_data.py ./data/processed_geom/dataset_geom_report0183.pkl ./data/processed_geom/ 40 60 80 100 120 
python sample_npz_data.py ./data/processed_graph/dataset_graph_report0183.npz ./data/processed_graph/ 40 60 80 100 120 

# ==============================
# SC experiments
# ==============================
# Process sample40
nohup python -u run_sc.py -t sample40 --gnn GIN --opt adam --n_fold 5 -s 42 --device 5 --results_dir ./results_sample > ./log_sample/sample40_sc.out &

# Process sample60
nohup python -u run_sc.py -t sample60 --gnn GIN --opt adam --n_fold 5 -s 42 --device 5 --results_dir ./results_sample > ./log_sample/sample60_sc.out &

# Process sample80
nohup python -u run_sc.py -t sample80 --gnn GIN --opt adam --n_fold 5 -s 42 --device 5 --results_dir ./results_sample > ./log_sample/sample80_sc.out &

# Process sample100
nohup python -u run_sc.py -t sample100 --gnn GIN --opt adam --n_fold 5 -s 42 --device 6 --results_dir ./results_sample > ./log_sample/sample100_sc.out &

# Process sample120
nohup python -u run_sc.py -t sample120 --gnn GIN --opt adam --n_fold 5 -s 42 --device 6 --results_dir ./results_sample > ./log_sample/sample120_sc.out &

# Wait for SC experiments to complete
wait
echo "All SC experiments completed!"

# ==============================
# TL experiments
# ==============================
# Process sample40
nohup python -u run_tl.py -c SMRT -t sample40 --gnn GIN --opt adam --n_fold 5 -s 42 --device 5 --results_dir ./results_sample > ./log_sample/sample40_tl.out &

# Process sample60
nohup python -u run_tl.py -c SMRT -t sample60 --gnn GIN --opt adam --n_fold 5 -s 42 --device 5 --results_dir ./results_sample > ./log_sample/sample60_tl.out &

# Process sample80
nohup python -u run_tl.py -c SMRT -t sample80 --gnn GIN --opt adam --n_fold 5 -s 42 --device 5 --results_dir ./results_sample > ./log_sample/sample80_tl.out &

# Process sample100
nohup python -u run_tl.py -c SMRT -t sample100 --gnn GIN --opt adam --n_fold 5 -s 42 --device 6 --results_dir ./results_sample > ./log_sample/sample100_tl.out &

# Process sample120
nohup python -u run_tl.py -c SMRT -t sample120 --gnn GIN --opt adam --n_fold 5 -s 42 --device 6 --results_dir ./results_sample > ./log_sample/sample120_tl.out &

# Wait for TL experiments to complete
wait
echo "All TL experiments completed!"

# ==============================
# TSTL experiments
# ==============================
# Process sample40
nohup python -u run_tstl.py -c SMRT -t sample40 --gnn GIN --opt adam --device 5 --n_folds 5 --results_dir ./results_sample > ./log_sample/sample40_tstl.out &

# Process sample60
nohup python -u run_tstl.py -c SMRT -t sample60 --gnn GIN --opt adam --device 5 --n_folds 5 --results_dir ./results_sample > ./log_sample/sample60_tstl.out &

# Process sample80
nohup python -u run_tstl.py -c SMRT -t sample80 --gnn GIN --opt adam --device 5 --n_folds 5 --results_dir ./results_sample > ./log_sample/sample80_tstl.out &

# Process sample100
nohup python -u run_tstl.py -c SMRT -t sample100 --gnn GIN --opt adam --device 6 --n_folds 5 --results_dir ./results_sample > ./log_sample/sample100_tstl.out &

# Process sample120
nohup python -u run_tstl.py -c SMRT -t sample120 --gnn GIN --opt adam --device 6 --n_folds 5 --results_dir ./results_sample > ./log_sample/sample120_tstl.out &

# Wait for all TSTL experiments to complete
wait
echo "All TSTL experiments completed!"

echo "All experiments (SC, TL, TSTL) completed!"