import os
import pandas as pd
import json
import random
import pickle
import numpy as np

from rdkit import Chem
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem, rdDepictor

from util import MANUALLY_REMOVED_SUBSETS, check_metadata

# Original configurations kept separate
ENCODER = {
	'conf_type': 'etkdgv3',
	'atom_type': {
		'C': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		'H': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		'O': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
		'N': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
		'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		'S': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
		'Cl': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
		'P': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
		'B': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
		'Br': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
		'I': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
	},
	'max_atom_num': 300,
	'min_atom_num': 10
}

def determine_thr(x):
	# x is the rt column of dataframe
	value_counts = x.value_counts().sort_index()
	thr = 0
	pre_single_rt = True
	count_single_rt = 0
	for rt, counts in value_counts.items():
		# print(rt, counts)
		if counts > 1:
			thr = rt
			pre_single_rt = False
			count_single_rt = 0
			
		elif counts == 1 and pre_single_rt == True:
			count_single_rt += 1
		
		else:
			pre_single_rt = True
			count_single_rt = 1
			
		if count_single_rt > 2: 
			return thr

def conformation_array(smiles, conf_type): 
	# convert smiles to molecule
	if conf_type == 'etkdg': 
		mol = Chem.MolFromSmiles(smiles)
		mol_from_smiles = Chem.AddHs(mol)
		AllChem.EmbedMolecule(mol_from_smiles)

	elif conf_type == 'etkdgv3': 
		mol = Chem.MolFromSmiles(smiles)
		mol_from_smiles = Chem.AddHs(mol)
		ps = AllChem.ETKDGv3()
		ps.randomSeed = 0xf00d
		AllChem.EmbedMolecule(mol_from_smiles, ps) 

	elif conf_type == '2d':
		mol = Chem.MolFromSmiles(smiles)
		mol_from_smiles = Chem.AddHs(mol)
		rdDepictor.Compute2DCoords(mol_from_smiles)

	elif conf_type == 'omega': 
		raise ValueError('OMEGA conformation will be supported soon. ')
	else:
		raise ValueError('Unsupported conformation type. {}'.format(conf_type))

	# get the x,y,z-coordinates of atoms
	try: 
		conf = mol_from_smiles.GetConformer()
	except:
		return False, None, None
	xyz_arr = conf.GetPositions()
	# center the x,y,z-coordinates
	centroid = np.mean(xyz_arr, axis=0)
	xyz_arr -= centroid
	
	# concatenate with atom attributes
	xyz_arr = xyz_arr.tolist()
	for i, atom in enumerate(mol_from_smiles.GetAtoms()):
		xyz_arr[i] += [atom.GetDegree()]
		xyz_arr[i] += [atom.GetExplicitValence()]
		xyz_arr[i] += [atom.GetMass()/100]
		xyz_arr[i] += [atom.GetFormalCharge()]
		xyz_arr[i] += [atom.GetNumImplicitHs()]
		xyz_arr[i] += [int(atom.GetIsAromatic())]
		xyz_arr[i] += [int(atom.IsInRing())]
	xyz_arr = np.array(xyz_arr)
	
	# get the atom types of atoms
	atom_type = [atom.GetSymbol() for atom in mol_from_smiles.GetAtoms()]
	return True, xyz_arr, atom_type

def to_pkl(df, save_path, smiles_col='smiles', label_col='rt', id_col='id'):
	data = []
	for _, row in df.iterrows():
		# mol array
		good_conf, xyz_arr, atom_type = conformation_array(
			smiles=row[smiles_col], 
			conf_type=ENCODER['conf_type']
		)
		
		if not good_conf:
			print(f'Cannot generate correct conformation: {row[smiles_col]} {row[id_col]}')
			continue
			
		if xyz_arr.shape[0] > ENCODER['max_atom_num']:
			print(f'Atomic number ({xyz_arr.shape[0]}) exceeds the limitation ({ENCODER["max_atom_num"]})')
			continue
		
		# Check for unsupported atoms
		unsupported_atom = next((atom for atom in set(atom_type) 
							   if atom not in ENCODER['atom_type']), None)
		if unsupported_atom:
			print(f'Unsupported atom type: {unsupported_atom} {row[id_col]}')
			continue

		atom_type_one_hot = np.array([ENCODER['atom_type'][atom] for atom in atom_type])
		mol_arr = np.concatenate([xyz_arr, atom_type_one_hot], axis=1)
		mol_arr = np.pad(
			mol_arr, 
			((0, ENCODER['max_atom_num'] - xyz_arr.shape[0]), (0, 0)), 
			constant_values=0
		)
		
		data.append({
			'title': row[id_col],
			'smiles': row[smiles_col],
			'mol': np.transpose(mol_arr, (1, 0)),
			'rt': row[label_col]
		})
	
	with open(save_path, 'wb') as f:
		pickle.dump(data, f)
	return data

def process_subset(subset, df, out_dir, seed):
	"""Process a single subset of data"""
	df['smiles'] = df['smiles.std'].apply(
		lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=True)
	)
	
	# Average RT values for same compounds
	avg_rt_df = df.groupby('smiles')['rt'].mean().reset_index()
	avg_rt_df['id'] = avg_rt_df.index

	# Filter out non-eluting compounds
	thr = determine_thr(avg_rt_df['rt'])
	avg_rt_df = avg_rt_df[avg_rt_df['rt'] > thr]
	
	return avg_rt_df 

def save_dataset(subset_id, df, out_dir, smiles_col='smiles', label_col='rt', id_col='id'):
	"""Save dataset to both CSV and PKL formats"""
	if subset_id == 'SMRT': 
		base_path = os.path.join(out_dir, f'dataset_geom_{subset_id}')
	else: 
		base_path = os.path.join(out_dir, f'dataset_geom_report{subset_id}')
	# df.to_csv(f'{base_path}.csv', index=False) # CSV format is not needed now 
	to_pkl(df, f'{base_path}.pkl', smiles_col=smiles_col, label_col=label_col, id_col=id_col)

def process_SMRT_dataset(smrt_path, threshold, out_dir):
	suppl = Chem.SDMolSupplier(smrt_path, removeHs=False)
	df = []
	for mol in suppl: 
		if mol is not None and mol.HasProp('RETENTION_TIME') and mol.HasProp('PUBCHEM_COMPOUND_CID'):  
			item = {}
			item['rt'] = mol.GetProp('RETENTION_TIME')
			item['id'] = mol.GetProp('PUBCHEM_COMPOUND_CID')
			smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
			item['smiles'] = smiles
			df.append(item)
	df = pd.DataFrame(df)
	df['rt'] = pd.to_numeric(df['rt'], errors='coerce') # Convert to numeric and mark failed conversions as NaN
	df = df.dropna(subset=['rt']) # Remove rows where conversion failed (became NaN)
	if len(df) < threshold:
		print(f"SMRT dataset has too few samples: {len(df)} < {threshold}")
	else:
		save_dataset('SMRT', df, out_dir)
		print('SMRT dataset is processed.')

def process_RepoRT_dataset(report_dir, threshold, out_dir):
	# Initialize tracking
	discarded_record = {
		'manually_removed': [],
		'missing_metadata': [],
		'missing_required_condition': [],
		'too_few_samples': [],
		'missing_rt_data': [],
		'explore': [],
	}
	sample_numbers = ""
	pretrained_dfs = {}
	
	# Process each subset
	subset_list = [f for f in os.listdir(report_dir) if os.path.isdir(os.path.join(report_dir, f))]
	for subset in subset_list:
		subset_id = subset.split('_')[0]
		
		# Filter 1: Check if subset should be excluded
		if subset_id in MANUALLY_REMOVED_SUBSETS:
			discarded_record['manually_removed'].append(subset)
			continue
		
		# Filter 2: check if the subset has the required columns in metadata
		metadata_path = os.path.join(report_dir, subset, "{}_metadata.tsv".format(subset))
		if not os.path.exists(metadata_path):
			print('Metadata not found: {}'.format(metadata_path))
			discarded_record['missing_metadata'].append(subset)
			continue
		if not check_metadata(metadata_path):
			print('Missing required columns: {}'.format(metadata_path))
			discarded_record['missing_required_condition'].append(subset)
			continue

		# Load data
		subset_path = os.path.join(report_dir, subset, f"{subset}_rtdata_isomeric_success.tsv")
		if not os.path.exists(subset_path):
			discarded_record['missing_rt_data'].append(subset)
			continue
			
		df = pd.read_csv(subset_path, sep='\t')
		avg_rt_df = process_subset(subset, df, out_dir, seed)
		sample_numbers += f"{subset_id}: {len(avg_rt_df)}\n"

		# Handle different subset types
		if len(avg_rt_df) >= threshold:
			save_dataset(subset_id, avg_rt_df, out_dir)
			discarded_record['explore'].append(subset)
		else:
			discarded_record['too_few_samples'].append(subset)
			continue
	
	# Save results
	with open(os.path.join(out_dir, "report_geom_discarded_record.json"), 'w') as f:
		json.dump(discarded_record, f, indent=4)
	
	# Save pretrain data
	for subset_id, df in pretrained_dfs.items(): 
		save_dataset(subset_id, df, out_dir)
	
	# Print summary
	print('='*20)
	print('RepoRT Dataset Summary:')
	for k, v in discarded_record.items():
		print(f'{k}: {len(v)}')
	print('='*20)

	# Write sample numbers to file
	with open(os.path.join(out_dir, "report_geom_sample_numbers.txt"), 'w') as f:
		f.write(sample_numbers)

if __name__ == "__main__": 
	# Configuration ==============================
	threshold = 100  # at least 10-shot for TSTL (70% for TL or TSTL, 30% for testing)
	seed = 42
	out_dir = "./data/processed_geom"

	# Setup
	if not os.path.exists("./data/RepoRT/processed_data"):
		raise FileNotFoundError("Report dataset not found. Please download the data first.")
	if not os.path.exists("./data/SMRT_dataset.sdf"):
		raise FileNotFoundError("SMRT dataset not found. Please download the data first.")
	os.makedirs(out_dir, exist_ok=True)
	random.seed(seed)
	
	# RepoRT ==============================
	process_RepoRT_dataset("./data/RepoRT/processed_data", threshold=threshold, out_dir=out_dir)

	# SMRT ==============================
	process_SMRT_dataset("./data/SMRT_dataset.sdf", threshold=threshold, out_dir=out_dir)