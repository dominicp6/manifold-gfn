from typing import List, Optional

import h5py
import dgl
import numpy.typing as npt
import torch
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from utils.constants import ad_atom_types
from utils.featurizer import MolDGLFeaturizer
from utils.rdkit_conformer import RDKitConformer
from utils.rotatable_bonds import get_torsion_angles, find_backbone_dihedrals

from hyper_torus import HyperTorus

from torchtyping import TensorType


def get_positions(smiles: str) -> npt.NDArray:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)

    conformer = mol.GetConformer()
    
    return conformer.GetPositions()


class ConformerEnvironment(HyperTorus):

    def __init__(
        self,
        device,
        float_precision,
        proxy,
        config,
        molecule_name: str,
        smiles: str,
    ):  
        boltzmann_constant = 3.1668114e-6 # hartree/K
        self.beta = 1.0 / (boltzmann_constant * config.general.T) # hartree^-1 
        self.molecule_name = molecule_name
        self.smiles = smiles
        print(f"beta: {self.beta}")
        
        # Assert that dataset is a hdf5 dataset
        # assert dataset.endswith(".hdf5"), "Dataset must be a hdf5 file"
        # f = h5py.File(dataset, "r")
        # molecule = list(f.keys())[config.env.molecule_id]
        # self.molecule_name = molecule
        # self.smiles = f[molecule]["smiles"][0]
        self.atom_positions = get_positions(self.smiles)
        self.torsion_angles = get_torsion_angles(self.smiles)
        self.backbone_torsions = find_backbone_dihedrals(self.smiles)
        if config.env.backbone_only:
            self.torsion_angles = self.backbone_torsions["phi"] + self.backbone_torsions["psi"] + self.backbone_torsions["omega1"] + self.backbone_torsions["omega2"]
        self.set_conformer()
        assert len(self.torsion_angles) > 0, f"No rotatable bonds found in molecule {self.smiles}"

        self.graph = MolDGLFeaturizer(ad_atom_types).mol2dgl(self.conformer.rdk_mol)
        
        # Add a feature to the edges to indicate whether they are rotatable or not
        # Note: Central two atoms in a 4-atom torsion angle correspond to a rotatable bond
        rotatable_edges = [torsion_angle[1:3] for torsion_angle in self.torsion_angles] 
        for i in range(self.graph.num_edges()):
            if (self.graph.edges()[0][i].item(), self.graph.edges()[1][i].item()) not in rotatable_edges:
                self.graph.edata["rotatable_edges"][i] = False

        # TODO: is this now redundant with the earlier remove hydrogens code
        # We remove hydrogen atoms since they are not considered in the torsion angles and are thus not part of the action space
        self.remove_hydrogens = config.env.remove_hydrogens
        self.hydrogens = torch.where(self.graph.ndata["atom_features"][:, 0] == 1)[0]
        self.non_hydrogens = torch.where(self.graph.ndata["atom_features"][:, 0] != 1)[0]
        if config.env.remove_hydrogens:
            self.graph = dgl.remove_nodes(self.graph, self.hydrogens)

        self.n_dim = len(self.conformer.freely_rotatable_tas)
        super().__init__(proxy=proxy, device=device, n_dim=self.n_dim, float_precision=float_precision, config=config)

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

    def sync_conformer_with_state(self, state):
        for idx, ta in enumerate(self.conformer.freely_rotatable_tas):
            self.conformer.set_torsion_angle(ta, state[idx])
        return self.conformer
    
    def statebatch2conformerbatch(self, states: TensorType["batch_size", "n_dim"]) -> tuple[TensorType["batch_size", "N_atoms"], TensorType["batch_size", "N_atoms", 3]]:
        """
        Converts a batch of states to a batch of conformers.

        Args:
        - states: Tensor with dimensionality (batch_size, n_dim), which encodes torsion angles for the batch of states.

        Returns:
        - atomic_numbers: Tensor with dimensionality (batch_size, N_atoms), which encodes atomic numbers for the batch of states.
        - conformers_coords: Tensor with dimensionality (batch_size, N_atoms, 3), which encodes atomic positions for the batch of states.
        """
        batch_size = states.shape[0]
        atomic_numbers = torch.zeros((batch_size, self.conformer.get_n_atoms()), dtype=torch.int32, device=self.device)
        conformers_coords = torch.zeros((batch_size, self.conformer.get_n_atoms(), 3), device=self.device)
        for conf_idx in range(states.shape[0]):
            conf = self.sync_conformer_with_state(states[conf_idx])
            conformers_coords[conf_idx] = torch.tensor(conf.get_atom_positions()) 
            atomic_numbers[conf_idx] = torch.tensor(conf.get_atomic_numbers())

        return atomic_numbers, conformers_coords
    
    def _compute_boltzman_weights(self, states):
        # TODO: check that this is still correct
        conformers = self.statebatch2conformerbatch(states)
        energies = self.proxy(*conformers).cpu().numpy()
        
        if self.config.proxy.normalise:
            energies = (energies + 1) * (self.proxy.max_energy - self.proxy.min_energy) 
        
        boltzmann_weights = np.exp( - energies * self.beta)

        return boltzmann_weights
    
    def _compute_free_energy_histogram(self, states, boltzmann_weights, num_bins_per_dimension):
        # TODO: replace np with torch?
        histograms = []
        for dim in range(self.n_dim):
            # Extract the angles for the current dimension
            dim_marginal_states = states[:, dim].cpu().numpy()

            # Compute the histogram and weights for the Boltzmann density
            hist, _ = np.histogram(dim_marginal_states, bins=num_bins_per_dimension, range=(0, 2 * np.pi), weights=boltzmann_weights)
            norm_counts, _ = np.histogram(dim_marginal_states, bins=num_bins_per_dimension, range=(0, 2 * np.pi))
            average_density = np.divide(hist, norm_counts, out=np.zeros_like(hist), where=norm_counts > 0)
            normalised_density = np.divide(average_density, np.sum(average_density))

            # Store results
            histograms.append(normalised_density)
        
        return histograms
    
