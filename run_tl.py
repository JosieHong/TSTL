from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import random_split
import os
from datetime import datetime

from util import set_random_seed, get_dataset_and_model, create_data_loader, setup_checkpoint_path
from util import evaluate_model, save_metrics_to_csv
from trainer import tlTrainer

def train_and_evaluate(model_name, task_name, root_dir, source_path, target_path, is_pretraining, 
						val_ratio=0.1, batch_size=32, lr=1e-4, opt='adam', method='finetune',
						fold_id=None, n_splits=5, device=0, seed=42, verbose=False): 
	device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
	
	# Get appropriate dataset and model
	train_dataset, test_dataset, model, collate_fn, train_labels, test_labels = \
		get_dataset_and_model(model_name=model_name, task_name=task_name, root_dir=root_dir, 
								fold_id=fold_id, n_splits=n_splits, seed=seed)
	print(f'-- trn/tst: {len(train_dataset)}/{len(test_dataset)}')
	
	# Calculate training statistics
	trn_y_mean, trn_y_std = np.mean(train_labels), np.std(train_labels)
	
	# Setup trainer
	if len(train_labels) > 2000: 
		batch_size *= 4 # Increase batch size for large datasets
		lr *= 2 # Increase learning rate for large datasets
	trainer = tlTrainer(model, device, config={'random_state': seed, 'adam_lr': lr})
	
	if is_pretraining:
		# Create validation split for pre-training
		val_size = int(np.round(val_ratio * len(train_dataset)))
		train_size = len(train_dataset) - val_size
		train_subset, val_subset = random_split(
			train_dataset,
			[train_size, val_size],
			generator=torch.Generator().manual_seed(seed) if seed else None
		)
		
		# Create data loaders for pre-training
		train_loader = create_data_loader(
			train_subset,
			batch_size=min(train_size, batch_size),
			shuffle=True,
			collate_fn=collate_fn
		)
		val_loader = create_data_loader(
			val_subset,
			batch_size=batch_size,
			shuffle=False,
			collate_fn=collate_fn
		)

		print('-- training...')
		trainer.train(
			train_loader,
			val_loader,
			trn_y_mean,
			trn_y_std,
			save_path=target_path,
			method='scratch',
			opt='adam',
			verbose=verbose,
		)
	else: 
		# Fine-tuning setup
		test_loader = create_data_loader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
		train_loader = create_data_loader(
			train_dataset,
			batch_size=batch_size if opt == 'adam' else None,
			shuffle=opt == 'adam',
			collate_fn=collate_fn
		)
		
		print('-- training...')
		trainer.train(
			train_loader,
			test_loader,
			trn_y_mean,
			trn_y_std,
			save_path=target_path,
			source_path=source_path,
			method=method,
			opt=opt,
			verbose=verbose,
		)
	
	# Evaluate model
	if target_path:
		trainer.load(target_path)
	test_loader = create_data_loader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
	test_predictions = trainer.inference(test_loader, trn_y_mean, trn_y_std)
	return evaluate_model(test_labels, test_predictions)

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--data_dir', '-r', type=str, default=None,
					   help='Directory containing the dataset') # Set default path in 'util.py:get_dataset_and_model'
	parser.add_argument('--source', '-c', type=str, default='SMRT',
					   help='Pre-training task')
	parser.add_argument('--target', '-t', type=str, default='report0063',
					   help='Fine-tuning task')
	parser.add_argument('--gnn', '-g', type=str, 
					   choices=['MPNN', 'GAT', 'GTN', 'GIN', 'GIN3', '3DMol'], 
					   default='GIN')
	parser.add_argument('--method', '-m', type=str, 
					   choices=['feature', 'finetune'], 
					   default='finetune')
	parser.add_argument('--opt', '-o', type=str, 
					   choices=['adam', 'lbfgs'], 
					   default='adam')
	parser.add_argument('--seed', '-s', type=int, default=42,
					   help='Random seed for reproducibility')
	parser.add_argument('--n_folds', type=int, default=1,
					   help='Number of folds for k-fold validation (1 means randomly split)')
	parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
					   help='Directory to save checkpoints') 
	parser.add_argument('--results_dir', type=str, default='./results',
					   help='Directory to save results')
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--verbose', action='store_true', 
					   help='Print detailed training information')
	args = parser.parse_args()
	
	os.makedirs(args.checkpoint_dir, exist_ok=True)
	os.makedirs(args.results_dir, exist_ok=True)
	
	set_random_seed(args.seed)

	assert (args.gnn == '3DMol' and args.opt == 'adam') or args.gnn != '3DMol', "3DMol only supports Adam optimizer"

	# Fixed configuration
	pre_batch_size = 128 # Batch size for pre-training
	pre_learning_rate = 1e-3 # Learning rate for pre-training
	batch_size = 32 # Batch size for fine-tuning
	learning_rate = 1e-4 # Learning rate for fine-tuning
	val_ratio = 0.1

	# =========================================
	# Pre-training stage
	# =========================================
	print("\n= PRE-TRAINING STAGE")

	# Setup checkpoint path for pretraining
	source_path = setup_checkpoint_path(model_name=args.gnn, 
										strategy='tl', 
										stage='pre', 
										source_dataset=args.source, 
										target_dataset=None, 
										fold_id=None, 
										base_dir=args.checkpoint_dir)
	print(f'- source_path: {source_path}')

	if os.path.exists(source_path):
		print(f'- found existing pretrained model at {source_path}')
		print('- skipping pre-training stage')
	else: 
		# Train and evaluate model for pre-training (8:2 train-val split)
		metrics = train_and_evaluate(model_name=args.gnn, 
									task_name=args.source, 
									root_dir=args.data_dir,
									source_path=None, 
									target_path=source_path,
									is_pretraining=True,
									val_ratio=val_ratio, 
									batch_size=pre_batch_size,
									lr=pre_learning_rate,  
									opt=args.opt, 
									method=args.method,
									fold_id=None, n_splits=5, device=args.device, seed=42, verbose=args.verbose)
		
		# Save pre-training metrics
		result_path = save_metrics_to_csv(metrics, 
										model_name=args.gnn, 
										strategy='tl', 
										method=args.method, 
										opt=args.opt, 
										stage='pre',  
										source_dataset=args.source, 
										target_dataset=None, 
										fold_id=None,
										results_dir=args.results_dir)
		print('- save pre-training results_path:', result_path)

		print('-' * 50)
		print(f'- pre-training results on {args.source}:')
		for metric_name, value in metrics.items():
			print(f'- {metric_name}: {value:.4f}')
		print('-' * 50, '\n')

	# =========================================
	# Fine-tuning stage
	# =========================================
	print("\n= FINE-TUNING STAGE")
	if args.n_folds > 1: 
		all_metrics = []
		for fold in range(args.n_folds):
			print(f'- Fold {fold + 1}/{args.n_folds}')
			# Setup checkpoint path for this fold
			target_path = setup_checkpoint_path(model_name=args.gnn, 
										strategy='tl', 
										stage='ft', 
										source_dataset=args.source, 
										target_dataset=args.target, 
										fold_id=fold, 
										base_dir=args.checkpoint_dir)
			print(f'-- source_path: {source_path}')
			print(f'-- target_path: {target_path}')
			
			# Train and evaluate model
			metrics = train_and_evaluate(model_name=args.gnn, 
									task_name=args.target, 
									root_dir=args.data_dir,
									source_path=source_path, 
									target_path=target_path,
									is_pretraining=False,
									val_ratio=val_ratio, 
									batch_size=batch_size, 
									lr=learning_rate,
									opt=args.opt, 
									method=args.method,
									fold_id=fold, n_splits=args.n_folds, device=args.device, seed=42, verbose=args.verbose)
			all_metrics.append(metrics)
			
			# Save metrics for this fold
			result_path = save_metrics_to_csv(metrics, 
												model_name=args.gnn, 
												strategy='tl', 
												method=args.method, 
												opt=args.opt, 
												stage='ft', 
												source_dataset=args.source, 
												target_dataset=args.target, 
												fold_id=fold,
												results_dir=args.results_dir)
			print('-- save all results_path:', result_path)

			# Print metrics for this fold
			print('-' * 50)
			print('- results:')
			for metric_name, value in metrics.items():
				print(f'- {metric_name}: {value:.4f}')
			print('-' * 50)
		
		# Calculate and save average metrics
		avg_metrics = {}
		std_metrics = {}
		for metric in all_metrics[0].keys():
			values = [m[metric] for m in all_metrics]
			avg_metrics[f'avg_{metric}'] = np.mean(values)
			std_metrics[f'std_{metric}'] = np.std(values)
		
		# Save average metrics
		result_path = save_metrics_to_csv(avg_metrics, 
											model_name=args.gnn, 
											strategy='tl', 
											method=args.method, 
											opt=args.opt, 
											stage='ft', 
											source_dataset=args.source, 
											target_dataset=args.target, 
											fold_id='mean',
											results_dir=args.results_dir)
		print('- save mean results_path:', result_path)
		result_path = save_metrics_to_csv(std_metrics,
											model_name=args.gnn, 
											strategy='tl', 
											method=args.method, 
											opt=args.opt, 
											stage='ft', 
											source_dataset=args.source, 
											target_dataset=args.target, 
											fold_id='std',
											results_dir=args.results_dir)
		print('- save std results_path:', result_path)

		print('-' * 50)
		print('- average results across all folds:')
		for metric in all_metrics[0].keys():
			values = [m[metric] for m in all_metrics]
			print(f'- average {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}')
		print('-' * 50, '\n')

	else: # Single split mode (8:2 train-test split)
		print('- Single split mode')
		target_path = setup_checkpoint_path(model_name=args.gnn,
										strategy='tl', 
										stage='ft', 
										source_dataset=args.source, 
										target_dataset=args.target, 
										fold_id=None, 
										base_dir=args.checkpoint_dir)
		print(f'-- source_path: {source_path}')
		print(f'-- target_path: {target_path}')
		
		# Train and evaluate model
		metrics = train_and_evaluate(model_name=args.gnn, 
									task_name=args.target, 
									root_dir=args.data_dir,
									source_path=source_path, 
									target_path=target_path,
									is_pretraining=False,
									val_ratio=val_ratio, 
									batch_size=batch_size, 
									lr=learning_rate,
									opt=args.opt, 
									method=args.method,
									fold_id=None, n_splits=5, device=args.device, seed=42, verbose=args.verbose)
		
		# Save metrics for single run
		result_path = save_metrics_to_csv(metrics, 
										model_name=args.gnn, 
										strategy='tl', 
										method=args.method, 
										opt=args.opt, 
										stage='ft', 
										source_dataset=args.source, 
										target_dataset=args.target, 
										fold_id=None,
										results_dir=args.results_dir)
		print('-- save all results_path:', result_path)
		
		print('-' * 50)
		print('- results:')
		for metric_name, value in metrics.items():
			print(f'- {metric_name}: {value:.4f}')
		print('-' * 50, '\n')
