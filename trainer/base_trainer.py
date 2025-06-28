from itertools import cycle
import numpy as np
import time
import copy
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau



class baseTrainer:
	def __init__(self, model, device, config={}):
		self.model = model.double().to(device)
		self.device = device
		self.random_state = config.get('random_state', 42)
		self._set_random_state()
		self.criterion = nn.HuberLoss() # Loss function 

	def _set_random_state(self):
		np.random.seed(self.random_state)
		
		torch.manual_seed(self.random_state)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(self.random_state)
			torch.cuda.manual_seed_all(self.random_state)
	
	def load(self, model_path, encoder_only=False): 
		model_state = torch.load(model_path, map_location=self.device, weights_only=True)
		if encoder_only:
			# Load only the encoder part of the model
			encoder_state = {
				k: v for k, v in model_state.items() if k.startswith('encoder.')
			}
			self.model.encoder.load_state_dict(encoder_state)
		else: 
			self.model.load_state_dict(model_state)

	def inference(self, data_loader, mean, std): 
		self.model.eval()
		with torch.no_grad():
			preds = [self.model(*[b.to(self.device) for b in batch[:-1]]).cpu().numpy() 
					for batch in data_loader]
		return (np.vstack(preds) * std + mean)