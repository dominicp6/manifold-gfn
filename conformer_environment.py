from typing import List, Optional, Tuple

import dgl
import numpy.typing as npt
import torch

from rdkit import Chem
from rdkit.Chem import AllChem

from utils.constants import ad_atom_types
from utils.featurizer import MolDGLFeaturizer
from utils.rdkit_conformer import RDKitConformer
from utils.rotatable_bonds import find_rotor_from_smiles

from hyper_torus import HyperTorus
from process_data import SMILESDataset

from torchtyping import TensorType


def get_positions(smiles: str) -> npt.NDArray:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    
    return mol.GetConformer().GetPositions()

def get_torsion_angles(smiles: str) -> List[Tuple[int]]:
    torsion_angles = find_rotor_from_smiles(smiles)
    
    return torsion_angles

class ConformerEnvironment(HyperTorus):

    def __init__(
        self,
        proxy,
        dataset: SMILESDataset,
        single_molecule:bool = True,
        molecule_id:int = 0,
        remove_hydrogens: bool = True,
        n_comp: int = 3,
        traj_length: int = 5,
        encoding_multiplier: int = 5,
        vonmises_min_concentration: float = 1e-3,
        float_precision: int = 32,
        device: str = "cpu"
    ):  
        if single_molecule:
            self.smiles = dataset.molecules[molecule_id]
            self.atom_positions = get_positions(self.smiles)
            self.torsion_angles = get_torsion_angles(self.smiles)
            self.set_conformer()
            assert len(self.torsion_angles) > 0, f"No rotatable bonds found in molecule {self.smiles}"

        self.graph = MolDGLFeaturizer(ad_atom_types).mol2dgl(self.conformer.rdk_mol)
        
        rotatable_edges = [ta[1:3] for ta in self.torsion_angles] # TODO: why ta[1:3]?
        for i in range(self.graph.num_edges()):
            if (self.graph.edges()[0][i].item(), self.graph.edges()[1][i].item()) not in rotatable_edges:
                self.graph.edata["rotatable_edges"][i] = False

        # We remove hydrogen atoms since they are not considered in the torsion angles and are thus not part of the action space
        self.remove_hydrogens = remove_hydrogens
        self.hydrogens = torch.where(self.graph.ndata["atom_features"][:, 0] == 1)[0]
        self.non_hydrogens = torch.where(self.graph.ndata["atom_features"][:, 0] != 1)[0]
        if remove_hydrogens:
            self.graph = dgl.remove_nodes(self.graph, self.hydrogens)

        self.n_dim = len(self.conformer.freely_rotatable_tas)
        super().__init__(proxy=proxy, device=device, vonmises_min_concentration=vonmises_min_concentration, n_dim=self.n_dim, n_comp=n_comp, traj_length=traj_length, encoding_multiplier=encoding_multiplier, float_precision=float_precision)
        self.sync_conformer_with_state()

        # Set up proxy
        self.proxy = proxy
        print("Setting up proxy")
        self.proxy.setup(self)
        print("Proxy set up")

    def set_conformer(self, state: Optional[List] = None) -> RDKitConformer:
        self.conformer = RDKitConformer(self.atom_positions, self.smiles, self.torsion_angles)

        if state is not None:
            self.sync_conformer_with_state(state)

        return self.conformer

    def sync_conformer_with_state(self, state = None):
        if state is None:
            state = self.state
        for idx, ta in enumerate(self.conformer.freely_rotatable_tas):
            self.conformer.set_torsion_angle(ta, state[idx])
        return self.conformer
    
    def statebatch2conformerbatch(self, states: TensorType["batchsize", "n_dim"]) -> tuple[TensorType["N_atoms"], TensorType["batchsize", "N_atoms", 3]]:
        """
        Returns two Tensors. The first is a list of atomic numbers for the conformer.
        The second is a Tensor with dimensionality (batchsize, N_atoms, 3), which encodes 
        atomic positions for the batch of states.
        """
        # TODO: replace states.shape[0] with a more readable reference to batchsize
        atomic_numbers = torch.zeros((states.shape[0], self.conformer.get_n_atoms()), dtype=torch.int32)
        conformers_coords = torch.zeros((states.shape[0], self.conformer.get_n_atoms(), 3))
        for conf_idx in range(states.shape[0]):
            conf = self.sync_conformer_with_state(states[conf_idx])
            conformers_coords[conf_idx] = torch.tensor(conf.get_atom_positions()) 
            atomic_numbers[conf_idx] = torch.tensor(conf.get_atomic_numbers())

        return atomic_numbers, conformers_coords
