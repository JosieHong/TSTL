import os
import pandas as pd
import numpy as np
import json
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures

from util import MANUALLY_REMOVED_SUBSETS, check_metadata

# Disable RDKit logging
rdBase.DisableLog('rdApp.*')

# Constants for feature extraction
ATOM_LIST = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
CHARGE_LIST = [1, -1, 0]
DEGREE_LIST = [1, 2, 3, 4, 0]
VALENCE_LIST = [1, 2, 3, 4, 5, 6, 0]
HYBRIDIZATION_LIST = ['SP', 'SP2', 'SP3', 'S']
HYDROGEN_LIST = [1, 2, 3, 0]
RINGSIZE_LIST = [3, 4, 5, 6, 7, 8]
BOND_LIST = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

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

class MolecularPreprocessor:
	def __init__(self, output_dir="./data", seed=42):
		self.output_dir = output_dir
		self.seed = seed
		self.chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(
			os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
		)
		os.makedirs(output_dir, exist_ok=True)

	def _get_DA_features(self, mol):
		"""Extract donor and acceptor features."""
		D_list, A_list = [], []
		for feat in self.chem_feature_factory.GetFeaturesForMol(mol):
			if feat.GetFamily() == 'Donor':
				D_list.append(feat.GetAtomIds()[0])
			if feat.GetFamily() == 'Acceptor':
				A_list.append(feat.GetAtomIds()[0])
		return D_list, A_list

	def _get_chirality(self, atom):
		"""Get chirality features for an atom."""
		if atom.HasProp('Chirality'):
			return [(atom.GetProp('Chirality') == 'Tet_CW'),
				   (atom.GetProp('Chirality') == 'Tet_CCW')]
		return [0, 0]

	def _get_stereochemistry(self, bond):
		"""Get stereochemistry features for a bond."""
		if bond.HasProp('Stereochemistry'):
			return [(bond.GetProp('Stereochemistry') == 'Bond_Cis'),
				   (bond.GetProp('Stereochemistry') == 'Bond_Trans')]
		return [0, 0]

	def _process_stereochemistry(self, mol):
		"""Process and set stereochemistry properties for a molecule."""
		si = Chem.FindPotentialStereo(mol)
		for element in si:
			if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
				mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
			elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
				mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
		return mol

	def _process_molecule(self, mol): 
		"""Process a single molecule to extract its features."""
		try:
			mol = self._process_stereochemistry(mol)
			assert '.' not in Chem.MolToSmiles(mol)
			mol = Chem.RemoveHs(mol)
		except:
			return None

		n_node = mol.GetNumAtoms()
		n_edge = mol.GetNumBonds() * 2
		
		D_list, A_list = self._get_DA_features(mol)
		
		# Atom features
		atom_features = {
			'symbol': np.eye(len(ATOM_LIST))[[ATOM_LIST.index(a.GetSymbol()) for a in mol.GetAtoms()]],
			'charge': np.eye(len(CHARGE_LIST))[[CHARGE_LIST.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1],
			'degree': np.eye(len(DEGREE_LIST))[[DEGREE_LIST.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1],
			'hybridization': np.eye(len(HYBRIDIZATION_LIST))[[HYBRIDIZATION_LIST.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-1],
			'hydrogen': np.eye(len(HYDROGEN_LIST))[[HYDROGEN_LIST.index(a.GetTotalNumHs(includeNeighbors=True)) for a in mol.GetAtoms()]][:,:-1],
			'valence': np.eye(len(VALENCE_LIST))[[VALENCE_LIST.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1],
			'DA': np.array([[(j in D_list), (j in A_list)] for j in range(n_node)]),
			'chirality': np.array([self._get_chirality(a) for a in mol.GetAtoms()]),
			'ring_size': np.array([[a.IsInRingSize(s) for s in RINGSIZE_LIST] for a in mol.GetAtoms()]),
			'aromatic': np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()])
		}
		
		node_attr = np.concatenate(list(atom_features.values()), axis=1)
		
		edge_attr, src, dst = None, None, None
		if n_edge > 0:
			bond_features = {
				'type': np.eye(len(BOND_LIST))[[BOND_LIST.index(str(b.GetBondType())) for b in mol.GetBonds()]],
				'stereo': np.array([self._get_stereochemistry(b) for b in mol.GetBonds()]),
				'properties': [[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()]
			}
			
			edge_attr = np.concatenate(list(bond_features.values()), axis=1)
			edge_attr = np.vstack([edge_attr, edge_attr])
			
			bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()])
			src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
			dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])
		
		return {
			'n_node': n_node,
			'n_edge': n_edge,
			'node_attr': node_attr,
			'edge_attr': edge_attr,
			'src': src,
			'dst': dst
		}

	def process_molecules(self, mol_list, label_list):
		"""Process a list of molecules and their labels."""
		processed_data = {
			'n_node': [], 'n_edge': [], 'node_attr': [], 
			'edge_attr': [], 'src': [], 'dst': [], 'label': []
		}
		
		for i, (mol, label) in enumerate(zip(mol_list, label_list)):
			mol_features = None
			try: 
				mol_features = self._process_molecule(mol)
			except Exception as e:
				print(f"Error processing molecule {i}: {str(e)}")
				continue

			if mol_features is None:
				continue
				
			for key, value in mol_features.items():
				if value is not None:
					processed_data[key].append(value)
			processed_data['label'].append(label)
			
			if (i + 1) % 10000 == 0:
				print(f'--- {i+1}/{len(mol_list)} processed')
		
		if len(processed_data['label']) == 0: # No valid molecules
			return processed_data

		# Convert lists to arrays
		for key in processed_data:
			if key in ['n_node', 'n_edge']:
				processed_data[key] = np.array(processed_data[key]).astype(int)
			elif key in ['node_attr', 'edge_attr']:
				processed_data[key] = np.vstack(processed_data[key]).astype(bool)
			elif key in ['src', 'dst']:
				processed_data[key] = np.hstack(processed_data[key]).astype(int)
			elif key == 'label':
				processed_data[key] = np.array(processed_data[key]).astype(float)
		
		return processed_data

	def process_SMRT_dataset(self, sdf_path):
		"""Process METLIN-SMRT dataset."""
		print("\nProcessing METLIN-SMRT dataset...")
		suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
		
		mol_list = []
		RT_list = []
		for mol in suppl:
			try:
				Chem.SanitizeMol(mol)
				RT = mol.GetDoubleProp('RETENTION_TIME')
				if RT > 200:
					mol_list.append(mol)
					RT_list.append(RT)
			except:
				continue
		
		processed_data = self.process_molecules(mol_list, RT_list)
		np.savez_compressed(os.path.join(self.output_dir, 'dataset_graph_SMRT.npz'),
						  data=[processed_data])
		
		print(f'METLIN-SMRT processed: {len(RT_list)} molecules')
		print(f'RT range: {min(RT_list):.1f} - {max(RT_list):.1f}')

	def process_RepoRT_dataset(self, root_dir, threshold): 
		"""Process RepoRT dataset."""
		print("\nProcessing RepoRT dataset...")
		discarded_record = {
			'manually_removed': [],
			'missing_metadata': [],
			'missing_required_condition': [],
			'too_few_samples': [],
			'missing_rt_data': [],
			'explore': []
		}
		sample_numbers = ""
		
		subset_list = [f for f in os.listdir(root_dir) 
					  if os.path.isdir(os.path.join(root_dir, f))]
		
		for subset in subset_list:
			subset_id = subset.split('_')[0]
			
			# Filter 1: Check if subset should be excluded
			if subset_id in MANUALLY_REMOVED_SUBSETS:
				discarded_record['manually_removed'].append(subset)
				continue
			
			# Filter 2: check if the subset has the required columns in metadata
			metadata_path = os.path.join(root_dir, subset, "{}_metadata.tsv".format(subset))
			if not os.path.exists(metadata_path):
				print('Metadata not found: {}'.format(metadata_path))
				discarded_record['missing_metadata'].append(subset)
				continue
			if not check_metadata(metadata_path):
				print('Missing required columns: {}'.format(metadata_path))
				discarded_record['missing_required_condition'].append(subset)
				continue

			subset_path = os.path.join(root_dir, subset, 
									 f"{subset}_rtdata_isomeric_success.tsv")
			if not os.path.exists(subset_path):
				discarded_record['missing_rt_data'].append(subset)
				continue
			
			# Read and process dataset
			df = pd.read_csv(subset_path, sep='\t')
			df['smiles'] = df['smiles.std'].apply(
				lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=True)
			)
			sample_numbers += f"{subset_id}: {len(df)}\n"
			
			mol_list = [Chem.MolFromSmiles(smiles) for smiles in df['smiles']]
			rt_list = df['rt'].tolist()

			# Filter out non-eluting compounds
			thr = determine_thr(df['rt'])
			df = df[df['rt'] > thr] 
			
			if len(df) < threshold: 
				discarded_record['too_few_samples'].append(subset)
				print('Too few samples: {}'.format(subset))
			else: 
				processed_data = self.process_molecules(mol_list, rt_list)
				np.savez_compressed(
					os.path.join(self.output_dir, f'dataset_graph_report{subset_id}.npz'),
					data=[processed_data]
				)
				discarded_record['explore'].append(subset)
				print(f'Processed {subset_id}: {len(processed_data["label"])}')

		# Save discarded record
		with open(os.path.join(self.output_dir, 'report_graph_discarded_record.json'), 'w') as f:
			json.dump(discarded_record, f, indent=4)
		
		# Write sample numbers to file
		with open(os.path.join(self.output_dir, 'report_graph_sample_numbers.txt'), 'w') as f:
			f.write(sample_numbers)

		print('='*20)
		print('RepoRT Dataset Summary:')
		for k, v in discarded_record.items():
			print(f'{k}: {len(v)}')
		print('='*20)



if __name__ == "__main__":
	# preprocessor = MolecularPreprocessor(output_dir="./data/processed_graph", seed=42)
	preprocessor = MolecularPreprocessor(output_dir="./tmp/", seed=42)

	# Process METLIN-SMRT dataset
	# preprocessor.process_SMRT_dataset('./data/SMRT_dataset.sdf')
	
	# Process RepoRT dataset
	preprocessor.process_RepoRT_dataset('./data/RepoRT/processed_data', threshold=100)