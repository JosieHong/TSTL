# Task-Specific Transfer Learning for Retention Time Prediction

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] (free for academic use) 

Task-Specific Transfer Learning (TSTL) is a transfer learning approach for liquid chromatography (LC) retention time prediction across diverse LC systems and experimental conditions with limited training data. Neural networks are pre-trained on distinct large-scale datasets, each optimized as an initialization point for specific target tasks, then fine-tuned and integrated to achieve superior performance on downstream tasks. 

Preprint: https://www.biorxiv.org/content/10.1101/2025.06.26.661631v1

## 🛠️ Set up

**Step 1**: Establish anaconda environment

```bash
conda create -n pre_gnn python=3.9
conda activate pre_gnn

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c dglteam/label/th24_cu124 dgl
pip install rdkit pandas matplotlib pydantic scikit-learn seaborn
```

**Step 2**: Download and preprocess SMRT and RepoRT datasets

```bash
bash get_data.sh 
```

Results of preprocessing: 
```bash
# graph
====================
RepoRT Dataset Summary:
manually_removed: 31
missing_metadata: 0
missing_required_condition: 159
too_few_samples: 193
missing_rt_data: 1
explore: 36
====================

# geom
====================
RepoRT Dataset Summary:
manually_removed: 31
missing_metadata: 0
missing_required_condition: 159
too_few_samples: 191
missing_rt_data: 1
explore: 38
====================
```



## 🚀 Quick start

In this section, `report0184` is used as an example to show the application of different training strategies (SC, TL, and TSTL) with different models (MPNN, GTN, GIN, GIN3, and 3DMol). 

### Training from scratch

```bash
python run_sc.py -t report0184 --n_fold 5 -s 42

# Also support randomly splitting the data into training and test sets.
# python run_sc.py -t report0184 -s 42
```

Other optional arguments: 
```bash
usage: run_sc.py [-h] [--data_dir DATA_DIR] [--target TARGET] [--gnn {MPNN,GAT,GTN,GIN,GIN3,3DMol}]
                 [--opt {adam,lbfgs}] [--seed SEED] [--n_folds N_FOLDS] [--device DEVICE]
                 [--checkpoint_dir CHECKPOINT_DIR] [--results_dir RESULTS_DIR] [--verbose]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR, -d DATA_DIR
                        Directory containing the dataset
  --target TARGET, -t TARGET
                        Fine-tuning task
  --gnn {MPNN,GAT,GTN,GIN,GIN3,3DMol}, -g {MPNN,GAT,GTN,GIN,GIN3,3DMol}
  --opt {adam,lbfgs}, -o {adam,lbfgs}
  --seed SEED, -s SEED  Random seed for reproducibility
  --n_folds N_FOLDS     Number of folds for k-fold validation (1 means randomly split)
  --device DEVICE
  --checkpoint_dir CHECKPOINT_DIR
                        Directory to save checkpoints
  --results_dir RESULTS_DIR
                        Directory to save results
  --verbose             Print detailed output during training
```

### Conventional transfer learning (TL)

```bash
python run_tl.py -c SMRT -t report0184 --n_fold 5 -s 42 

# Also support randomly splitting the data into training and test sets.
# python run_tl.py -c SMRT -t report0184 -s 42 --verbose --opt adam
```

Other optional arguments: 
```bash
usage: run_tl.py [-h] [--data_dir DATA_DIR] [--source SOURCE] [--target TARGET] [--gnn {MPNN,GAT,GTN,GIN,GIN3,3DMol}]
                 [--method {feature,finetune}] [--opt {adam,lbfgs}] [--seed SEED] [--n_folds N_FOLDS]
                 [--checkpoint_dir CHECKPOINT_DIR] [--results_dir RESULTS_DIR] [--device DEVICE] [--verbose]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR, -r DATA_DIR
                        Directory containing the dataset
  --source SOURCE, -c SOURCE
                        Pre-training task
  --target TARGET, -t TARGET
                        Fine-tuning task
  --gnn {MPNN,GAT,GTN,GIN,GIN3,3DMol}, -g {MPNN,GAT,GTN,GIN,GIN3,3DMol}
  --method {feature,finetune}, -m {feature,finetune}
  --opt {adam,lbfgs}, -o {adam,lbfgs}
  --seed SEED, -s SEED  Random seed for reproducibility
  --n_folds N_FOLDS     Number of folds for k-fold validation (1 means randomly split)
  --checkpoint_dir CHECKPOINT_DIR
                        Directory to save checkpoints
  --results_dir RESULTS_DIR
                        Directory to save results
  --device DEVICE
  --verbose             Print detailed training information
```

### Task-Specific Transfer Learning (TSTL)

```bash
python run_tstl.py -c SMRT report0390 report0391 report0392 report0262 report0383 report0382 report0263 -t report0184 --n_fold 5 -s 42 

# Also support randomly splitting the data into training and test sets.
# python run_tstl.py -c SMRT report0390 report0391 report0392 report0262 report0383 report0382 report0263 -t report0184 -s 42
```

Other optional arguments: 
```bash
usage: run_tstl.py [-h] [--data_dir DATA_DIR] [--sources SOURCES [SOURCES ...]] [--target TARGET]
                   [--gnn {MPNN,GAT,GTN,GIN,GIN3,3DMol}] [--method {feature,finetune}] [--opt {adam,lbfgs}]
                   [--n_folds N_FOLDS] [--seed SEED] [--checkpoint_dir CHECKPOINT_DIR] [--results_dir RESULTS_DIR]
                   [--device DEVICE] [--verbose] [--integrate_only]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory containing the dataset
  --sources SOURCES [SOURCES ...], -c SOURCES [SOURCES ...]
  --target TARGET, -t TARGET
  --gnn {MPNN,GAT,GTN,GIN,GIN3,3DMol}, -g {MPNN,GAT,GTN,GIN,GIN3,3DMol}
  --method {feature,finetune}, -m {feature,finetune}
  --opt {adam,lbfgs}, -o {adam,lbfgs}
  --n_folds N_FOLDS     Number of folds for k-fold validation (1 means randomly split)
  --seed SEED, -s SEED
  --checkpoint_dir CHECKPOINT_DIR
                        Directory to save checkpoints
  --results_dir RESULTS_DIR
                        Directory to save results
  --device DEVICE
  --verbose             Print detailed training information
  --integrate_only      Only integrate models without training
```



## 🔬 Experiments

### 1. define TL-difficult datasets

Running conventional transfer learning (TL) on all datasets with a 8:2 random split, and picking the datasets with R2 < 0.8 as **TL-difficult** datasets. GIN is utilized here according to the work from [Kwon et al.](https://pubs.acs.org/doi/10.1021/acs.analchem.3c03177). 

```bash
# training from scratch
nohup ./experiments_run_gin_sc_adam.sh > experiments_run_gin_sc_adam.log 2>&1 &
nohup ./experiments_run_gin_sc_lbfgs.sh > experiments_run_gin_sc_lbfgs.log 2>&1 &

# training with transfer learning
nohup ./experiments_run_gin_tl_adam.sh > experiments_run_gin_tl_adam.log 2>&1 &
nohup ./experiments_run_gin_tl_lbfgs.sh > experiments_run_gin_tl_lbfgs.log 2>&1 &
```

### 2. run TSTL on TL-difficult datasets

Running task specific transfer learning on TL-difficult datasets with 5-fold validation. All the baseline models are utilized here. 

```bash
nohup bash experiments_run_all_tldiff.sh > experiments_run_all_tldiff.out 2>&1 &
```

### 3. does TSTL decreased the requirement of training size? 

```bash
nohup bash experiments_run_sample.sh > experiments_run_sample.out & 
```



## Citation

TBA

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
