import os
import yaml
import torch
import dgl
import numpy as np
import pandas as pd
import random
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score
import csv
from torch.utils.data import DataLoader

from model import MPNNPredictor, GINPredictor, GTNPredictor, GATPredictor, MolNet_RT
from dataset import GraphDataset, GeomDataset

# =========================================
# For preprocessing the dataset
# =========================================
# RepoRT dataset specific constants
MANUALLY_REMOVED_SUBSETS = [
	# manually discarded 23 datasets
	"0006", "0008", "0023", "0059", "0123", "0128", 
	"0130", "0131", "0132", "0133", "0136", "0137", 
	"0139", "0143", "0145", "0148", "0149", "0151", 
	"0152", "0154", "0155", "0156", "0157",
	# manually discarded two datasets
	"0056", "0057", 
	# One dataset uses a step-wise gradient
	"0024",
	# Suspicious subsets in meta-data
	"0004", "0005", "0015", "0021", "0041"
]

REQUIRED_COLS = [
	# (essential) Column properties
	"id", "column.name", "column.t0", "column.length", "column.particle.size", 
	
	# (essential) Temperature
	"column.temperature", 
	
	# (essential) Flow rate
	"column.flowrate", 

	# # pH
	# "eluent.A.pH", "eluent.B.pH", "eluent.C.pH", "eluent.D.pH",

	# # Gradient
	# "gradient.start.A", "gradient.start.B", "gradient.start.C", "gradient.start.D", 
	# "gradient.end.A", "gradient.end.B", "gradient.end.C", "gradient.end.D",
]

def check_metadata(path):
	df = pd.read_csv(path, sep='\t')
	for col in REQUIRED_COLS:
		if col not in df.columns:
			print(f"Missing required column: {col}") 
			return False
		if df[col].isnull().any():
			print(f"Column {col} has null values.") 
			return False
	return True

# =========================================
# For training and evaluation
# =========================================
def set_random_seed(seed): 
	# Set Python's random seed
	random.seed(seed)
	
	# Set NumPy's random seed
	np.random.seed(seed)
	
	# Set PyTorch's random seed
	torch.manual_seed(seed)

	dgl.seed(seed)

def collate_2D_graphs(batch): 
	g_list, label_list = map(list, zip(*batch))
	
	g_list = dgl.batch(g_list)
	label_list = torch.FloatTensor(np.vstack(label_list))
	
	return g_list, g_list.ndata['node_attr'], g_list.edata['edge_attr'], label_list

def setup_checkpoint_path(model_name, strategy, stage, 
							source_dataset, target_dataset, 
							fold_id=None, base_dir='./checkpoint'):
	# Create checkpoint directory if it doesn't exist
	if not os.path.exists(base_dir): 
		os.makedirs(base_dir)
	
	# Validate strategy and required parameters
	# assert strategy in ['sc', 'tl', 'tstl'], "Strategy must be 'sc', 'tl', or 'tstl'."
	assert stage in ['pre', 'ft', 'warmup', None], "Stage must be 'pre', 'ft', 'warmup', or None."
	assert (strategy == 'sc' and stage == None and source_dataset == None) \
		   or (strategy == 'warmup' and target_dataset == None) \
			or (strategy != 'sc' or strategy != 'warmup'), "Stage and source should be None for sc strategy. Target should be None for warmup strategy."
	if stage == None: stage = 'nan'
	if source_dataset == None: source_dataset = 'nan'
	if target_dataset == None: target_dataset = 'nan'
	
	if fold_id is not None:
		return f"{base_dir}/{model_name}_{strategy}_{stage}_{source_dataset}_{target_dataset}_kfold{fold_id}.pt" # k-fold cross-validation
	else:
		return f"{base_dir}/{model_name}_{strategy}_{stage}_{source_dataset}_{target_dataset}.pt" # randomly spliting

def create_model(model_type, node_feats, edge_feats): 
	if model_type == 'GIN':
		return GINPredictor(node_in_feats=node_feats, edge_in_feats=edge_feats)
	elif model_type == 'GIN3':
		return GINPredictor(node_in_feats=node_feats, edge_in_feats=edge_feats, num_layers=3)
	elif model_type == 'MPNN':
		return MPNNPredictor(node_in_feats=node_feats, edge_in_feats=edge_feats)
	elif model_type == 'GAT':
		return GATPredictor(node_in_feats=node_feats, edge_in_feats=edge_feats)
	elif model_type == 'GTN':
		return GTNPredictor(node_in_feats=node_feats, edge_in_feats=edge_feats)
	else:
		raise ValueError(f"Unknown model type: {model_type}. Supported types are 'GIN', 'GIN3', 'MPNN', 'GAT', and 'GTN'.")

def create_data_loader(dataset, batch_size=None, shuffle=False, collate_fn=None):
	return DataLoader(
		dataset=dataset,
		batch_size=batch_size or len(dataset),
		collate_fn=collate_fn,
		shuffle=shuffle,
		drop_last=shuffle
	)

def get_dataset_and_model(model_name, task_name, root_dir=None, fold_id=None, n_splits=5, seed=42): 
	if root_dir is None: 
		root_dir = './data/processed_graph' if model_name != '3DMol' else './data/processed_geom'

	if model_name != '3DMol':
		train_dataset = GraphDataset(
			root_dir=root_dir,
			name=task_name, 
			cv_id=0 if fold_id==None else fold_id,
			n_splits=n_splits,
			split='trn', 
			seed=seed
		)
		test_dataset = GraphDataset(
			root_dir=root_dir,
			name=task_name, 
			cv_id=0 if fold_id==None else fold_id,
			n_splits=n_splits,
			split='tst',
			seed=seed
		)
		collate_fn = collate_2D_graphs
		
		model = create_model(
			model_name,
			train_dataset.node_attr.shape[1],
			train_dataset.edge_attr.shape[1]
		)
	else: # 3DMol specific setup
		train_dataset = GeomDataset(
			root_dir=root_dir,
			name=task_name, 
			cv_id=0 if fold_id==None else fold_id,
			n_splits=n_splits,
			split='trn', 
			seed=seed
		)
		test_dataset = GeomDataset(
			root_dir=root_dir,
			name=task_name, 
			cv_id=0 if fold_id==None else fold_id, 
			n_splits=n_splits,
			split='tst', 
			seed=seed
		)
		collate_fn = None
		
		# Load MolNet configuration and create model
		with open('./model/molnet.yml', 'r') as f:
			config = yaml.load(f, Loader=yaml.FullLoader)
		model = MolNet_RT(config['model'])
		
	train_labels = train_dataset.label
	test_labels = test_dataset.label

	return train_dataset, test_dataset, model, collate_fn, train_labels, test_labels

def evaluate_model(y_true, y_pred): 
	return {
		'MAE': mean_absolute_error(y_true, y_pred),
		'MedAE': np.median(np.abs(y_true - y_pred)),
		'R2': r2_score(y_true, y_pred)
	}

def save_metrics_to_csv(metrics, model_name, strategy, method, opt, stage, source_dataset, target_dataset, fold_id, results_dir): 
	# assert strategy in ['sc', 'tl', 'tstl'], "Invalid strategy for saving metrics"
	assert stage in [None, 'pre', 'ft', 'ensemble'], "Invalid stage for saving metrics"

	os.makedirs(results_dir, exist_ok=True)
	filename = os.path.join(results_dir, f'./metrics_{model_name}_{strategy}_{target_dataset}.csv')

	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	row_data = {
		'timestamp': timestamp,
		'model_name': model_name,
		'stage': stage if stage is not None else '-',
		'source_dataset': source_dataset if stage == 'pre' or stage == 'ft' else '-', 
		'target_dataset': target_dataset,
		'ft_method': method if stage == 'ft' else '-',
		'optimizer': opt,
		'fold': fold_id if fold_id is not None else '-',
		**metrics
	}
	
	file_exists = os.path.isfile(filename)
	
	with open(filename, 'a', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=row_data.keys())
		if not file_exists:
			writer.writeheader()
		writer.writerow(row_data)
	
	return filename
