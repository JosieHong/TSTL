from itertools import cycle
import numpy as np
import time
import copy
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .base_trainer import baseTrainer

class tlTrainer(baseTrainer):
	def __init__(self, net, cuda, config={}):
		super().__init__(net, cuda, config)
		defaults = {
			'adam_lr': 1e-4,
			'adam_weight_decay': 0.01,
			
			'lbfgs_lr': 1,
			'lbfgs_max_iter': 1, 

			'early_stopping_patience': 30, 
		}
		self.config = {**defaults, **config}

	def _setup_transfer_learning(self, method, source_path):
		if method == 'scratch':
			return
		if not source_path:
			raise ValueError('Source path required for fine-tuning or feature extraction')
			
		self.load(source_path)
		if method == 'feature': 
			for param in self.model.encoder.parameters():
				param.requires_grad = False
				
		if method in ['feature', 'finetune']:
			for layer in self.model.decoder:
				if hasattr(layer, 'reset_parameters'):
					layer.reset_parameters()

	def _train_adam(self, train_loader, val_loader, loss_fn, mean, std, save_path, max_epochs, verbose):
		# optimizer = Adam(self.model.parameters(), 
		# 				lr=self.config['adam_lr'], 
		# 				weight_decay=self.config['adam_weight_decay'])
		optimizer = AdamW(self.model.parameters(), 
						lr=self.config['adam_lr'], 
						weight_decay=self.config['adam_weight_decay'])
		scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
									patience=20, min_lr=1e-6)

		train_size = len(train_loader.dataset)
		val_size = len(val_loader.dataset)
		dataset = val_loader.dataset
		val_y = dataset.label if hasattr(dataset, 'label') \
					else dataset.dataset.label[dataset.indices]

		val_log = np.zeros(max_epochs)
		for epoch in range(max_epochs):
			# Training
			self.model.train()
			start = time.time()

			for batch in train_loader:
				inputs = [b.to(self.device) for b in batch[:-1]]
				labels = ((batch[-1] - mean) / std).to(self.device, dtype=torch.float64)
				
				preds = self.model(*inputs)
				loss = loss_fn(preds, labels)
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				
				train_loss = loss.detach().item()

			if verbose:
				print(f'--- training epoch {epoch + 1}, lr {optimizer.param_groups[-1]["lr"]:.6f}, '
					  f'processed {train_size}, loss {train_loss:.3f}, '
					  f'time elapsed(min) {(time.time()-start)/60:.2f}')

			# Validation
			start = time.time()
			val_preds = self.inference(val_loader, mean, std)
			val_loss = loss_fn(torch.FloatTensor((val_preds - mean)/std), 
							 torch.FloatTensor((val_y - mean)/std)).detach().cpu().numpy()
			
			scheduler.step(val_loss)
			val_log[epoch] = val_loss

			if verbose:
				print(f'--- validation at epoch {epoch + 1}, processed {val_size}, '
					  f'loss {val_loss:.3f} (BEST {np.min(val_log[:epoch + 1]):.3f}), '
					  f'monitor {epoch - np.argmin(val_log[:epoch + 1])}, '
					  f'time elapsed(min) {(time.time()-start)/60:.2f}')

			if np.argmin(val_log[:epoch + 1]) == epoch:
				torch.save(self.model.state_dict(), save_path)
				# print(f'--- Saving model at epoch {epoch + 1}', save_path)
			elif np.argmin(val_log[:epoch + 1]) <= epoch - self.config['early_stopping_patience']:
				break

		print(f'--- Training terminated at epoch {epoch + 1}')
		self.load(save_path) # Make sure self.model is the best model 

	def _train_lbfgs(self, train_loader, loss_fn, mean, std, save_path, max_epochs):
		for batch in train_loader:
			inputs = [b.to(self.device) for b in batch[:-1]]
			labels = ((batch[-1] - mean) / std).to(self.device, dtype=torch.float64)
			break

		def closure():
			optimizer.zero_grad()
			preds = self.model(*inputs)
			loss = loss_fn(preds, labels)
			loss += 1e-5 * sum(p.square().sum() for p in self.model.parameters())
			loss.backward()
			return loss

		optimizer = LBFGS(self.model.parameters(), 
						 lr=self.config['lbfgs_lr'], 
						 max_iter=self.config['lbfgs_max_iter'])
		
		val_log = np.zeros(max_epochs)
		for epoch in range(max_epochs):
			self.model.train()
			optimizer.step(closure)
			
			val_log[epoch] = closure().detach().cpu().numpy()
			val_log[epoch] = 1e5 if np.isnan(val_log[epoch]) else val_log[epoch]
			
			optimizer.param_groups[0]['lr'] -= 1/max_epochs

			if np.argmin(val_log[:epoch + 1]) == epoch:
				torch.save(self.model.state_dict(), save_path)
				# print(f'--- Saving model at epoch {epoch + 1}', save_path)

		print(f'-- Training terminated at iter {np.argmin(val_log) + 1}')
		self.load(save_path) # Make sure self.model is the best model

	def train(self, train_loader, val_loader, mean, std, method, opt, 
			  save_path, source_path='', max_epochs=500, verbose=False):
		self._setup_transfer_learning(method, source_path)
		
		loss_fn = nn.HuberLoss()
		if opt == 'adam': 
			self._train_adam(train_loader, val_loader, loss_fn, mean, std, 
						   save_path, max_epochs, verbose)
		elif opt == 'lbfgs':
			self._train_lbfgs(train_loader, loss_fn, mean, std, 
							save_path, max_epochs) 
		else:
			raise ValueError(f'Unsupported optimizer: {opt}')
