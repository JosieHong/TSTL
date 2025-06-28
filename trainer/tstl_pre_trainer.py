from itertools import cycle
import numpy as np
import time
import copy
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .base_trainer import baseTrainer

class tstlTrainer(baseTrainer):
	def __init__(self, net, cuda, config={}): 
		super().__init__(net, cuda, config)
		default = {
			'inner_lr': 1e-3,
			'meta_lr': 1e-4,
			'num_inner_steps': 10,
			'meta_weight_decay': 0.01,
			'early_stopping_patience': 10,

			'warmup_lr': 1e-4,
			'warmup_weight_decay': 0.01,
			'warmup_epochs': 10
		}
		self.config = {**default, **config}

		# Meta-optimizer setup
		self.meta_optimizer = AdamW(
			self.model.parameters(),
			lr=self.config['meta_lr'],
			weight_decay=self.config['meta_weight_decay']
		)
		self.meta_scheduler = ReduceLROnPlateau(
			self.meta_optimizer,
			mode='min',
			factor=0.1,
			patience=5,
			min_lr=1e-6
		)

	def load_encoder(self, warmup_save_path): 
		self.model.encoder.load_state_dict(torch.load(warmup_save_path, map_location=self.device, weights_only=True)) 

	def warmup_encoder(self, warmup_save_path, warmup_loader, warmup_val_loader, 
						mean, std): # Used to warm up the encoder parameters for a general molecular representation
		warmup_optimizer = AdamW(
			self.model.parameters(),
			lr=self.config['warmup_lr'],
			weight_decay=self.config['warmup_weight_decay'], 
		)
		
		for epoch in range(self.config['warmup_epochs']):
			warmup_losses = []
			self.model.train()
			
			for batch_data in warmup_loader:
				inputs = [b.to(self.device) for b in batch_data[:-1]]
				labels = ((batch_data[-1] - mean) / std).to(self.device, dtype=torch.float64)
				
				warmup_optimizer.zero_grad()
				preds = self.model(*inputs)
				loss = self.criterion(preds, labels)
				loss.backward()
				warmup_optimizer.step()
				
				warmup_losses.append(loss.item())
			
			avg_warmup_loss = np.mean(warmup_losses)	
			print(f"--- warmup epoch {epoch + 1} - loss {avg_warmup_loss:.4f}")

			val_losses = []
			self.model.eval()
			with torch.no_grad():
				for val_data in warmup_val_loader:
					val_inputs = [b.to(self.device) for b in val_data[:-1]]
					val_labels = ((val_data[-1] - mean) / std).to(self.device, dtype=torch.float64)

					val_preds = self.model(*val_inputs)
					val_loss = self.criterion(val_preds, val_labels)
					val_losses.append(val_loss.item())
			avg_val_loss = np.mean(val_losses)
			print(f"--- warmup epoch {epoch + 1} - val loss {avg_val_loss:.4f}")

		# Save the encoder parameters after warmup
		torch.save(self.model.encoder.state_dict(), warmup_save_path)
		print('--- saved warmup encoder parameters to %s' % warmup_save_path)

		# Reset decoder parameters after warmup
		for module in self.model.decoder.modules(): 
			if hasattr(module, 'reset_parameters'): 
				module.reset_parameters()
		print('--- reset decoder parameters after warmup')

	def _inner_loop(self, support_data, support_scaler, query_data=None, query_scaler=None, verbose=False): 
		# Process support data
		inputs, labels = support_data[:-1], support_data[-1]
		inputs = [b.to(self.device) for b in inputs]
		labels = ((labels - support_scaler['mean']) / support_scaler['std']).to(self.device, dtype=torch.float64)
		
		# Create local copy of model for inner loop
		local_model = copy.deepcopy(self.model)
		for param in self.model.encoder.parameters(): param.requires_grad = False # Freeze the encoder parameters for faster training
		local_optimizer = AdamW(local_model.parameters(), lr=self.config['inner_lr'])
		
		# Inner loop training on support set
		for step in range(self.config['num_inner_steps']):
			preds = local_model(*inputs)
			support_loss = self.criterion(preds, labels)
			
			local_optimizer.zero_grad()
			support_loss.backward()
			local_optimizer.step()
			
			if verbose:
				print('---- inner loop step %d, support loss %.4f' % (step + 1, support_loss.item()))
		
		# If query data is provided, calculate the loss on it
		if query_data is not None:
			query_inputs = [b.to(self.device) for b in query_data[:-1]]
			query_labels = ((query_data[-1] - query_scaler['mean']) / query_scaler['std']).to(self.device, dtype=torch.float64)
			
			query_preds = local_model(*query_inputs)
			query_loss = self.criterion(query_preds, query_labels)
			
			return query_loss, support_loss.item()
		
		return support_loss.item()

	def _meta_train_step(self, support_batch, query_batch, support_scaler, query_scaler): 
		self.meta_optimizer.zero_grad()
		
		meta_loss, _ = self._inner_loop(support_batch, support_scaler, query_data=query_batch, query_scaler=query_scaler)
		
		meta_loss.backward()
		self.meta_optimizer.step()
		self.meta_scheduler.step(meta_loss.item())

		return meta_loss.item()

	def train(self, support_loader, query_loader, val_loader, support_scaler, query_scaler, save_path, 
				max_epochs=500, verbose=False): 
		# Get dataset sizes
		if hasattr(val_loader.dataset, 'label'):
			support_size = support_loader.dataset.__len__()
			query_size = query_loader.dataset.__len__()
			val_size = val_loader.dataset.__len__()
		else:
			support_size = support_loader.dataset.indices.__len__()
			query_size = query_loader.dataset.indices.__len__()
			val_size = val_loader.dataset.indices.__len__()
		
		val_log = np.zeros(max_epochs)
		best_val_loss = float('inf')
		patience_counter = 0
		
		for epoch in range(max_epochs): 
			# Training
			self.model.train()
			start_time = time.time()
			epoch_losses = []
			
			# Iterate through support and query loaders simultaneously
			cyclic_support_loader = cycle(support_loader) # Create an infinite cycle of the query loader
			for query_batch in query_loader:
				support_batch = next(cyclic_support_loader)

				# Perform meta-training step
				meta_loss = self._meta_train_step(support_batch=support_batch, support_scaler=support_scaler, 
												query_batch=query_batch, query_scaler=query_scaler)
				epoch_losses.append(meta_loss)
			
			avg_train_loss = np.mean(epoch_losses)
			
			if verbose:
				print('--- epoch %d'%(epoch + 1))
				print('--- training, lr %f, support size %d, query size %d, query loss %.3f, time elapsed(min) %.2f'
					%(self.meta_optimizer.param_groups[-1]['lr'], 
					  support_size, query_size, avg_train_loss, (time.time()-start_time)/60))
			
			# Validation
			start_time = time.time()
			val_loss = []
			for val_data in val_loader: 
				val_loss.append(self._inner_loop(val_data, support_scaler, verbose=verbose))
			val_loss = np.mean(val_loss)
			val_log[epoch] = val_loss

			if verbose:
				print('--- validation, processed %d, support loss %.3f (BEST %.3f), time elapsed(min) %.2f'
					%(val_size, val_loss, np.min(val_log[:epoch + 1]), 
					  (time.time()-start_time)/60))
			
			# Save best model and check early stopping
			if val_loss < best_val_loss: 
				best_val_loss = val_loss
				torch.save(self.model.state_dict(), save_path)
				patience_counter = 0
				print('--- patience counter reset!')
			else:
				patience_counter += 1
				print('--- patience counter %d/%d' % (patience_counter, self.config['early_stopping_patience']))
				if patience_counter >= self.config['early_stopping_patience']:  # Early stopping patience
					break

		print('training terminated at epoch %d' %(epoch + 1))
		self.load(save_path) # Make sure self.model is the best model 