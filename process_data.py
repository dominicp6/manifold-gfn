import h5py
import pandas as pd
import numpy as np


class SMILESDataset:

    def __init__(self, data_npy_path: str):
        self.molecules = np.load(data_npy_path, allow_pickle=True)

    def get_random_molecule(self):
        return np.random.choice(self.molecules)
    

def parse_database_file_into_data_frame(path_to_db):
    # Initialize an empty list to collect data
    molecule_data = []
    print(path_to_db)

    # Open the HDF5 file
    with h5py.File(path_to_db, "r") as f:
        print("Loading file")
        print(f.keys())
        for molecule_id in f.keys():
            molecule_group = f[molecule_id]
            
            # Retrieve SMILES string
            smiles = molecule_group['smiles'][0] if 'smiles' in molecule_group else None
            
            # Retrieve conformations (if they exist)
            if 'conformations' in molecule_group:
                conformations = molecule_group['conformations'][:]
                num_conformations, num_atoms, _ = conformations.shape

                # Add each conformation separately
                for i in range(num_conformations):
                    molecule_data.append({
                        'id': molecule_id,
                        'smiles': smiles,
                        'conformation_index': i,
                        'num_atoms': num_atoms,
                        'positions': conformations[i]
                    })
            else:
                # If there are no conformations, add molecule data without positions
                molecule_data.append({
                    'id': molecule_id,
                    'smiles': smiles,
                    'conformation_index': None,
                    'num_atoms': None,
                    'positions': None
                })

    return pd.DataFrame(molecule_data)
    
def create_conformer_dataset_from_data_frame(df):
    #  Save a list of smiles strings
    smiles_strings = df['smiles'].unique()
    smiles_strings = np.array(smiles_strings)
    # Save as a pkl file for later use
    np.save("smiles_strings.npy", smiles_strings)
