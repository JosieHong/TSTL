import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
import copy
from dgl import graph
import pickle as pkl
from sklearn.model_selection import KFold

class GraphDataset(Dataset):
	def __init__(self, root_dir, name='SMRT', cv_id=0, n_splits=5, split='trn', seed=134):
		self.n_splits = n_splits 
		assert cv_id in list(range(self.n_splits)), f'cv_id ({cv_id}) should be in {range(self.n_splits)}'
		assert split in ['trn', 'tst', 'all']
		self.name = name
		self.cv_id = cv_id
		self.split = split
		self.seed = seed
		self.root_dir = root_dir
		self.load()

	def load(self): 
		[mol_dict] = np.load(os.path.join(self.root_dir, './dataset_graph_%s.npz'%self.name), allow_pickle=True)['data']
		kf = KFold(n_splits=self.n_splits, random_state=self.seed, shuffle=True)
		cv_splits = [split for split in kf.split(range(len(mol_dict['label'])))]
		cv_splits = cv_splits[self.cv_id]
		
		if self.split == 'trn': 
			mol_indices = np.array([i in cv_splits[0] for i in range(len(mol_dict['label']))], dtype=bool)
		elif self.split == 'tst':
			mol_indices = np.array([i in cv_splits[1] for i in range(len(mol_dict['label']))], dtype=bool)
		elif self.split == 'all':
			mol_indices = np.array([True for i in range(len(mol_dict['label']))], dtype=bool)

		node_indices = np.repeat(mol_indices, mol_dict['n_node'])
		self.label = mol_dict['label'][mol_indices].reshape(-1,1)

		edge_indices = np.repeat(mol_indices, mol_dict['n_edge'])
		self.n_node = mol_dict['n_node'][mol_indices]
		self.n_edge = mol_dict['n_edge'][mol_indices]
		self.node_attr = mol_dict['node_attr'][node_indices]
		self.edge_attr = mol_dict['edge_attr'][edge_indices]
		self.src = mol_dict['src'][edge_indices]
		self.dst = mol_dict['dst'][edge_indices]

		self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
		self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])
			
		assert len(self.n_node) == len(self.label)
			
	def __getitem__(self, idx):
		g = graph((self.src[self.e_csum[idx]:self.e_csum[idx+1]], self.dst[self.e_csum[idx]:self.e_csum[idx+1]]), num_nodes = self.n_node[idx])
		g.ndata['node_attr'] = torch.from_numpy(self.node_attr[self.n_csum[idx]:self.n_csum[idx+1]]).double()
		g.edata['edge_attr'] = torch.from_numpy(self.edge_attr[self.e_csum[idx]:self.e_csum[idx+1]]).double()
		label = self.label[idx].astype(np.float64)
		return g, label
		
	def __len__(self):
		return len(self.label)

class GeomDataset(Dataset): 
	def __init__(self, root_dir, name, cv_id=0, n_splits=5, split='trn', seed=134): 
		self.n_splits = n_splits 
		assert cv_id in list(range(self.n_splits))
		assert split in ['trn', 'tst', 'all']
		self.cv_id = cv_id
		self.split = split
		self.seed = seed

		with open(os.path.join(root_dir, './dataset_geom_%s.pkl'%name), 'rb') as file: 
			data = pkl.load(file)
		
		# load for different purposes
		kf = KFold(n_splits=self.n_splits, random_state=self.seed, shuffle=True)
		cv_splits = [split for split in kf.split(range(len(data)))]
		cv_splits = cv_splits[self.cv_id]
		
		if self.split == 'trn': 
			mol_indices = np.array([i in cv_splits[0] for i in range(len(data))], dtype=bool)
		elif self.split == 'tst':
			mol_indices = np.array([i in cv_splits[1] for i in range(len(data))], dtype=bool)
		elif self.split == 'all':
			mol_indices = np.array([True for i in range(len(data))], dtype=bool)
		
		self.data = [data[i] for i in range(len(data)) if mol_indices[i]]
		self.label = np.array([d['rt'] for d in self.data]).reshape(-1,1)
		
		# Generate mask
		for idx in range(len(self.data)): 
			mask = ~np.all(self.data[idx]['mol'] == 0, axis=0)
			self.data[idx]['mask'] = mask.astype(bool)

		# Add noise
		for i in range(3): # fixed configuration
			for idx in range(len(self.data)): 
				new_data = copy.deepcopy(self.data[idx])
				new_data['mol'] = self._add_noise(new_data['mol'], new_data['mask'])
				self.data.append(new_data)
		
	def _add_noise(self, mol, mask): 
		num_atoms = np.sum(mask)
		num_noisy_atoms = max(1, int(0.2 * num_atoms)) # Try this! 
		noisy_indices = np.random.choice(num_atoms, num_noisy_atoms, replace=False)
		xyz_corr = mol[:3, noisy_indices]
		xyz_corr += np.random.normal(0, 0.02, xyz_corr.shape)
		mol[:3, noisy_indices] = xyz_corr
		return mol

	def __len__(self): 
		return len(self.label)

	def __getitem__(self, idx): 
		return (self.data[idx]['mol'], 
				self.data[idx]['mask'], 
				self.label[idx])