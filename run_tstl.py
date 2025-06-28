from argparse import ArgumentParser
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple, Any, Union

from dataset import GraphDataset, GeomDataset
from util import set_random_seed, setup_checkpoint_path, get_dataset_and_model, create_data_loader
from util import evaluate_model, save_metrics_to_csv
from trainer import tstlTrainer, tstlIntegrator
from run_tl import train_and_evaluate

def tstl_pretrain_model(model_name, upstream_task, downstream_task, 
						data_dir, model_path, warmup_save_path, 
						fold_id, batch_size=32, device=0, seed=42, verbose=False):
	device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu') 
	if fold_id is None: 
		fold_id = 0
	
	# Load datasets using the helper function
	que_data, que_val_data, model, collate_fn, que_labels, que_val_labels = \
		get_dataset_and_model(model_name=model_name, task_name=upstream_task, root_dir=data_dir, fold_id=fold_id, seed=seed)
	sup_data, sup_val_data, _, _, sup_labels, sup_val_labels = \
		get_dataset_and_model(model_name=model_name, task_name=downstream_task, root_dir=data_dir, fold_id=fold_id, seed=seed)
	print(f'-- que/que_val/sup/sup_val: {len(que_data)}/{len(que_val_data)}/{len(sup_data)}/{len(sup_val_labels)}')

	# Create data loaders
	que_loader = create_data_loader(que_data, batch_size=min(batch_size, len(que_data)),
								  	shuffle=True, collate_fn=collate_fn)
	que_val_loader = create_data_loader(que_val_data, batch_size=min(batch_size, len(que_val_data)), 
									collate_fn=collate_fn)
	sup_loader = create_data_loader(sup_data, batch_size=min(batch_size, len(sup_data)),
								  	shuffle=True, collate_fn=collate_fn)
	sup_val_loader = create_data_loader(sup_val_data, batch_size=min(batch_size, len(sup_val_data)), 
									collate_fn=collate_fn)

	# Calculate statistics
	que_stats = {'mean': np.mean(que_labels), 'std': np.std(que_labels)}
	sup_stats = {'mean': np.mean(sup_labels), 'std': np.std(sup_labels)}
	
	# Train and evaluate
	trainer = tstlTrainer(model, device, config={'random_state': seed})

	# optional: Warmup
	if warmup_save_path and os.path.exists(warmup_save_path): 
		print(f'-- found existing warmup model at {warmup_save_path}')
		trainer.load_encoder(warmup_save_path)
	elif warmup_save_path: 
		print('-- warmup...')
		warmup_data, warmup_val_data, _, _, warmup_labels, warmup_val_labels = \
			get_dataset_and_model(model_name=model_name, task_name='SMRT', root_dir=data_dir, fold_id=None, seed=seed)
		print(f'-- warmup (SMRT): {len(warmup_data)}/{len(warmup_val_data)}')
		warmup_loader = create_data_loader(warmup_data, batch_size=512,
									shuffle=True, collate_fn=collate_fn)
		warmup_val_loader = create_data_loader(warmup_val_data, batch_size=512, 
									collate_fn=collate_fn)
		warmup_stats = {'mean': np.mean(warmup_labels), 'std': np.std(warmup_labels)}
		trainer.warmup_encoder(
			warmup_save_path, warmup_loader, warmup_val_loader, 
			warmup_stats['mean'], warmup_stats['std'],
		) 
	else: 
		print('-- no warmup path provided, skipping warmup stage')

	print('-- pre-training...')
	trainer.train(
		sup_loader, que_loader, que_val_loader,
		support_scaler=sup_stats,
		query_scaler=que_stats,
		save_path=model_path, 
		max_epochs=100,
		verbose=verbose,
	)
	
	# Evaluate
	trainer.load(model_path)

	predictions = trainer.inference(que_val_loader, sup_stats['mean'], sup_stats['std'])
	que_metrics = evaluate_model(que_val_labels, predictions)

	predictions = trainer.inference(sup_val_loader, que_stats['mean'], que_stats['std'])
	sup_metrics = evaluate_model(sup_val_labels, predictions)

	return que_metrics, sup_metrics

def create_and_evaluate_soup(model_name, downstream_task, target_paths, model_path, data_dir,
								batch_size=32, opt='adam', fold_id=None, device=0, seed=42): 
	device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
	
	# Get appropriate dataset and model
	train_dataset, test_dataset, model, collate_fn, train_labels, test_labels = \
		get_dataset_and_model(model_name=model_name, task_name=downstream_task, root_dir=data_dir, fold_id=None, seed=42)
	print(f'-- trn/tst: {len(train_dataset)}/{len(test_dataset)}')

	trn_y_mean, trn_y_std = np.mean(train_labels), np.std(train_labels)

	train_loader = create_data_loader(train_dataset, batch_size=batch_size if opt == 'adam' else None, 
									shuffle=False, collate_fn=collate_fn)
	test_loader = create_data_loader(test_dataset, batch_size=batch_size, 
									shuffle=False, collate_fn=collate_fn)

	integrator = tstlIntegrator(model, device=device)
	metrics = integrator.create_model_soup(
		checkpoint_paths=target_paths,
		train_loader=train_loader,
		valid_loader=test_loader,
		scaler={'mean': trn_y_mean, 'std': trn_y_std}, 
		ensemble_path=model_path, 
	)
	
	return metrics

def run_experiment(sources, target, gnn, method, opt, n_folds, 
				  batch_size, val_ratio, device, seed, verbose, 
				  data_dir, checkpoint_dir, results_dir, integrate_only=False):
	"""Main function to run the experiment with or without folds"""
	if n_folds > 1:
		for fold in range(n_folds):
			run_fold(sources, target, gnn, method, opt, fold, n_folds, 
					batch_size, val_ratio, device, seed, verbose, 
					data_dir, checkpoint_dir, results_dir, integrate_only)
	else:
		run_fold(sources, target, gnn, method, opt, None, 5, 
				batch_size, val_ratio, device, seed, verbose, 
				data_dir, checkpoint_dir, results_dir, integrate_only)

def run_fold(sources, target, gnn, method, opt, fold_id, n_folds, 
			batch_size, val_ratio, device, seed, verbose, 
			data_dir, checkpoint_dir, results_dir, integrate_only=False): 
	"""Run a single fold of the experiment"""

	# =========================================
	# Warmup stage (always use SMRT dataset)
	# =========================================
	print("\n= WARMUP STAGE")
	warmup_save_path = setup_checkpoint_path(model_name=args.gnn, 
											strategy='tstl', 
											stage='warmup', 
											source_dataset='SMRT', 
											target_dataset=None, 
											fold_id=None, 
											base_dir=args.checkpoint_dir)
	print(f'- warmup_save_path: {warmup_save_path}')

	# Print fold information
	if fold_id is not None:
		print(f"\nProcessing Fold {fold_id + 1}/{n_folds}")

		# =========================================
		# Pre-training stage
		# =========================================
		print("\n= PRE-TRAINING STAGE")
		source_paths = []
		
		for source in sources:
			# Setup checkpoint path for pretraining
			source_path = setup_checkpoint_path(
				model_name=gnn, 
				strategy='tstl', 
				stage='pre', 
				source_dataset=source, 
				target_dataset=target, 
				fold_id=fold_id, 
				base_dir=checkpoint_dir
			)
			print(f'- source_path: {source_path}')
			source_paths.append(source_path)
		
			# Pretrain the model
			if os.path.exists(source_path): 
				print(f'- found existing pretrained model at {source_path}')
				print('- skipping pre-training stage')
			elif not integrate_only:  
				que_metrics, sup_metrics = tstl_pretrain_model(
					gnn, source, target, data_dir, source_path, warmup_save_path, fold_id=fold_id, 
					batch_size=batch_size, device=device, seed=seed, verbose=verbose
				)
				
				# Print metrics
				print('-' * 50)
				print(f'- task specific pre-training results on {source} (* without fine-tuning):')
				for metric_name, value in que_metrics.items(): 
					print(f'- {metric_name}: {value:.4f}')
				print(f'- task specific pre-training results on {target} (* without fine-tuning):')
				for metric_name, value in sup_metrics.items(): 
					print(f'- {metric_name}: {value:.4f}')
				print('-' * 50)
		
		# =========================================
		# Fine-tuning stage
		# =========================================
		print("\n= FINE-TUNING STAGE")
		target_paths = []
		
		for source, source_path in zip(sources, source_paths):
			target_path = setup_checkpoint_path(
				model_name=gnn, 
				strategy='tstl', 
				stage='ft', 
				source_dataset=source, 
				target_dataset=target, 
				fold_id=fold_id, 
				base_dir=checkpoint_dir, 
			)
			print(f'-- source_path: {source_path}')
			print(f'-- target_path: {target_path}')
			target_paths.append(target_path)
			
			if not integrate_only:
				metrics = train_and_evaluate(
					model_name=gnn, 
					task_name=target, 
					root_dir=data_dir,
					source_path=source_path, 
					target_path=target_path, 
					is_pretraining=False, 
					val_ratio=val_ratio, 
					batch_size=batch_size, 
					opt=opt, 
					method=method,
					fold_id=fold_id, 
					n_splits=n_folds,
					device=device, 
					seed=seed, 
					verbose=verbose
				)
				
				# Save metrics
				result_path = save_metrics_to_csv(
					metrics, 
					model_name=gnn, 
					strategy='tstl', 
					method=method, 
					opt=opt, 
					stage='ft', 
					source_dataset=source, 
					target_dataset=target, 
					fold_id=fold_id,
					results_dir=results_dir, 
				)
				print(f'-- save all results (results_path: {result_path})')
				
				# Print metrics
				print('\n' + '-' * 50)
				print(f'- results ({source} -> {target}):')
				for metric_name, value in metrics.items():
					print(f'- {metric_name}: {value:.4f}')
				print('-' * 50)
	
	# Create and evaluate model soup
	ensemble_path = setup_checkpoint_path(
		model_name=gnn,
		strategy='tstl', 
		stage='ft', 
		source_dataset='ensemble', 
		target_dataset=target, 
		fold_id=fold_id, 
		base_dir=checkpoint_dir, 
	)
	print(f'-- target_paths: {target_paths}')
	print(f'-- ensemble_path: {ensemble_path}')
	
	soup_metrics = create_and_evaluate_soup(
		model_name=gnn, 
		downstream_task=target, 
		data_dir=data_dir,
		target_paths=target_paths, 
		model_path=ensemble_path, 
		batch_size=batch_size, 
		opt=opt, 
		fold_id=fold_id, 
		device=device, 
		seed=seed
	)
	
	result_path = save_metrics_to_csv(
		soup_metrics, 
		model_name=gnn, 
		strategy='tstl', 
		method=method, 
		opt=opt, 
		stage='ft', 
		source_dataset='ensemble', 
		target_dataset=target, 
		fold_id=fold_id,
		results_dir=results_dir,
	)
	print(f'-- save ensemble results (results_path: {result_path})')
	
	# Print metrics
	print('\n' + '-' * 50)
	print(f'- results (ensemble -> {target}):')
	for metric_name, value in soup_metrics.items():
		print(f'- {metric_name}: {value:.4f}')
	print('-' * 50)

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('--data_dir', type=str, default=None,
					   help='Directory containing the dataset') # Set default path in 'util.py:get_dataset_and_model'
	parser.add_argument('--sources', '-c', nargs='+', 
					   default=['SMRT', 'report0390', 'report0391'])
	parser.add_argument('--target', '-t', type=str, default='report0063')
	parser.add_argument('--gnn', '-g', type=str, 
					   choices=['MPNN','GAT','GTN','GIN','GIN3','3DMol'], 
					   default='GIN')
	parser.add_argument('--method', '-m', type=str, 
					   choices=['feature', 'finetune'], default='finetune')
	parser.add_argument('--opt', '-o', type=str, choices=['adam', 'lbfgs'], 
					   default='adam')
	parser.add_argument('--n_folds', type=int, default=1,
					   help='Number of folds for k-fold validation (1 means randomly split)')
	parser.add_argument('--seed', '-s', type=int, default=42)
	parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
					   help='Directory to save checkpoints') 
	parser.add_argument('--results_dir', type=str, default='./results',
					   help='Directory to save results')
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--verbose', action='store_true', 
					   help='Print detailed training information')
	parser.add_argument('--integrate_only', action='store_true',
					   help='Only integrate models without training')
	args = parser.parse_args()
	
	os.makedirs(args.checkpoint_dir, exist_ok=True)
	os.makedirs(args.results_dir, exist_ok=True)
	
	set_random_seed(args.seed)

	assert (args.gnn == '3DMol' and args.opt == 'adam') or args.gnn != '3DMol', "3DMol only supports Adam optimizer"

	# Fixed configuration
	batch_size = 32
	val_ratio = 0.1

	# =========================================
	# Pre-training and finetuning stage
	# =========================================
	print("\n= PRE-TRAINING ADN FINETUNING STAGE")
	# Run the experiment
	run_experiment(
		sources=args.sources,
		target=args.target,
		gnn=args.gnn,
		method=args.method,
		opt=args.opt,
		n_folds=args.n_folds,
		batch_size=batch_size,
		val_ratio=val_ratio,
		device=args.device,
		seed=args.seed,
		verbose=args.verbose,
		data_dir=args.data_dir,
		checkpoint_dir=args.checkpoint_dir,
		results_dir=args.results_dir,
		integrate_only=args.integrate_only, 
	)