import os
import random
import json
from datetime import datetime

import py3Dmol
import wandb
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import numpy as np

from alanine_dipeptide import AlanineDipeptide
from proxy import TorchANIMoleculeEnergy
from policy import MLP_Policy, Uniform_Policy
from gflownet import GFlowNet



def set_seed(seed = None):
   if seed is None:
      # Get a random seed from the current clock time
      seed = int(datetime.now().timestamp())
   
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)

class Config:
   def __init__(self, dictionary):
      for key, value in dictionary.items():
         if isinstance(value, dict):
            value = Config(value)  # Recursively turn nested dicts into Config
         setattr(self, key, value)

   def to_dict(self):
      """
      Recursively convert the Config object back into a dictionary.
      """
      config_dict = {}
      for key, value in self.__dict__.items():
         if isinstance(value, Config):
               config_dict[key] = value.to_dict()  # Recurse into nested Configs
         else:
               config_dict[key] = value
      return config_dict

class Experiment:
      
      def __init__(self, exp_name="debug_run", checkpoint=None, config=None, seed=None):
         assert checkpoint is not None or config is not None, "Either a checkpoint or a config must be provided."

         set_seed(seed)
         self.config_dict = config
         self.exp_name = exp_name
         self.using_checkpoint = checkpoint is not None
         self.checkpoint_file = None
         
         if config is not None:
            self._initialize_from_config()
         else:
            self._initialize_from_checkpoint(checkpoint)

         print(f"Initialised molecule {self.env.smiles}")
         print(f"Number of torsion angles: {len(self.env.torsion_angles)}")
         print(f"Encoding multiplier: {self.config.env.encoding_multiplier}")
         print(f"Mixture dimension: {self.config.env.n_comp}")

      def save_config(self):
         # Save the config to a file
         with open(f"{self.gfn.logging_dir}/config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=4)
            
      def _initialize_from_config(self):
         print("Initialising new experiment using provided config")
         self.config = Config(self.config_dict)
         
         self.device = self.config.general.device
         self.float_precision = self.config.general.float_precision

         self.proxy = TorchANIMoleculeEnergy(device=self.device, 
                                             float_precision=self.float_precision, 
                                             config=self.config.proxy)
         if self.config.env.molecule_name == "alanine_dipeptide":
            self.env = AlanineDipeptide(self.device, self.float_precision, self.proxy, self.config)
         else:
            raise ValueError(f"Unknown molecule name {self.config.env.molecule_name}")
         self.forward_policy = self._create_policy(self.config.forward_policy)
         self.backward_policy = self._create_backward_policy(self.config.backward_policy)
         self.gfn = GFlowNet(device=self.device,
                             float_precision=self.float_precision,
                              env=self.env,
                              forward_policy=self.forward_policy,
                              backward_policy=self.backward_policy,
                              config=self.config)
         
      def _initialize_from_checkpoint(self, checkpoint):
         print("Initialising new experiment using provided checkpoint")
         self.checkpoint_file = torch.load(checkpoint)
         
         self.config = Config(wandb.config)
         self.device = self.config.general.device
         self.float_precision = self.config.general.float_precision
         
         self.proxy = TorchANIMoleculeEnergy(device=self.device, 
                                             float_precision=self.float_precision, 
                                             config=self.config.proxy)
         if self.config.env.molecule_name == "alanine_dipeptide":
            self.env = AlanineDipeptide(self.device, self.float_precision, self.proxy, self.config)
         else:
            raise ValueError(f"Unknown molecule name {self.config.env.molecule_name}")
         self.forward_policy = self._create_policy(self.config.forward_policy)
         self.forward_policy.load_state_dict(self.checkpoint_file["forward_policy"])
         
         self.backward_policy = self._create_backward_policy(self.config.backward_policy, self.checkpoint_file)
         
         self.gfn = GFlowNet(device=self.device,
                             float_precision=self.float_precision,
                              env=self.env,
                              forward_policy=self.forward_policy,
                              backward_policy=self.backward_policy,
                              config=self.config)
         self.logZ = self.checkpoint_file["logZ"]
         self.gfn.opt.load_state_dict(self.checkpoint_file["optimizer"])
         self.gfn.lr_scheduler.load_state_dict(self.checkpoint_file["lr_scheduler"])
         self.buffer = self.checkpoint_file["buffer"]
         self.gfn.it = self.checkpoint_file["it"]

         # Append to an info file if it exists
         if os.path.exists(f"{self.gfn.logging_dir}/info.txt"):
            with open(f"{self.gfn.logging_dir}/info.txt", "a") as f:
               f.write(f"Initialised from checkpoint {checkpoint}\n")
               f.write(f"Starting from step {self.gfn.it}\n")   
         else:
            with open(f"{self.gfn.logging_dir}/info.txt", "w") as f:
               f.write(f"Initialised from checkpoint {checkpoint}\n")
               f.write(f"Starting from step {self.gfn.it}\n")
      
      def _create_policy(self, policy_config):
         return MLP_Policy(env=self.env, 
                           device=self.device, 
                           float_precision=self.float_precision, 
                           n_hid=policy_config.n_hid, 
                           n_layers=policy_config.n_layers)

      def _create_backward_policy(self, backward_config, checkpoint_file=None):
         if backward_config.uniform:
            return Uniform_Policy(env=self.env, 
                                    device=self.device, 
                                    float_precision=self.float_precision)
         else:
            backward_policy = MLP_Policy(env=self.env,
                                          device=self.device, 
                                          float_precision=self.float_precision, 
                                          n_hid=backward_config.n_hid, 
                                          n_layers=backward_config.n_layers)
            if checkpoint_file is not None:
               backward_policy.load_state_dict(checkpoint_file["backward_policy"])
            
            return backward_policy
      
      def train(self, n_train_steps):
         if not self.using_checkpoint:
            wandb.init(project=self.exp_name, config=self.config_dict)
         else:
            wandb.init(project=self.exp_name, config=self.checkpoint_file["config"])

         if self.gfn.logging_dir is None:
            if self.config.logging.checkpoints:
                self.gfn.logging_dir = f"{self.config.logging.log_directory}/{wandb.run.name}"
                os.makedirs(self.gfn.logging_dir)
         self.save_config()
         self.gfn.train(n_train_steps=n_train_steps)

      def visualise_molecule_3D(self):
         smiles = self.env.smiles
         torsion_angles = self.env.torsion_angles
         mol = Chem.MolFromSmiles(smiles)
         mol = Chem.AddHs(mol)
         AllChem.EmbedMolecule(mol, randomSeed=0)
         
         torsion_indices = []
         for ta in torsion_angles:
            atom1, atom2, atom3, atom4 = ta
            torsion_indices.append((atom1, atom2, atom3, atom4))

         # Get 3D conformer and prepare mol block
         conf = mol.GetConformer()
         mol_block = Chem.MolToMolBlock(mol)

         # Create 3D visualization
         viewer = py3Dmol.view(width=800, height=400)
         viewer.addModel(mol_block, "mol")
         viewer.setStyle({"stick": {}})

         # Highlight torsion angles
         for (atom1, atom2, atom3, atom4) in torsion_indices:
            viewer.addLine({"start": {"x": conf.GetAtomPosition(atom1).x, 
                                       "y": conf.GetAtomPosition(atom1).y, 
                                       "z": conf.GetAtomPosition(atom1).z},
                              "end": {"x": conf.GetAtomPosition(atom2).x, 
                                    "y": conf.GetAtomPosition(atom2).y, 
                                    "z": conf.GetAtomPosition(atom2).z},
                              "color": "red", "linewidth": 3})
            viewer.addLine({"start": {"x": conf.GetAtomPosition(atom3).x, 
                                       "y": conf.GetAtomPosition(atom3).y, 
                                       "z": conf.GetAtomPosition(atom3).z},
                              "end": {"x": conf.GetAtomPosition(atom4).x, 
                                    "y": conf.GetAtomPosition(atom4).y, 
                                    "z": conf.GetAtomPosition(atom4).z},
                              "color": "blue", "linewidth": 3})

         viewer.zoomTo()
         return viewer
      
      # def visualise_molecule_2D(self):
      #    smiles = self.env.smiles
      #    torsion_angles = self.env.torsion_angles
      #    mol = Chem.MolFromSmiles(smiles)
      #    mol = Chem.AddHs(mol)
      #    AllChem.EmbedMolecule(mol)

      #    # Annotate torsion atoms
      #    for idx, (atom1, atom2, atom3, atom4) in enumerate(torsion_angles):
      #       mol.GetAtomWithIdx(atom1).SetProp("atomNote", f"T{idx}_A1")
      #       mol.GetAtomWithIdx(atom2).SetProp("atomNote", f"T{idx}_A2")
      #       mol.GetAtomWithIdx(atom3).SetProp("atomNote", f"T{idx}_A3")
      #       mol.GetAtomWithIdx(atom4).SetProp("atomNote", f"T{idx}_A4")

      #    # Draw molecule
      #    return Draw.MolToImage(mol, size=(800, 800))

      def visualise_molecule_2D(self):
         smiles = self.env.smiles
         backbone_torsion_angles = self.env.backbone_torsions  #{f"{item}":[(item)] for item in self.env.torsion_angles}
         mol = Chem.MolFromSmiles(smiles)
         mol = Chem.AddHs(mol)
         AllChem.EmbedMolecule(mol)

         # Create a dictionary to store labels for each atom
         atom_labels = {}

         # Collect all torsion angle labels for each atom
         for angle_type, angles in backbone_torsion_angles.items():
            for idx, (atom1, atom2, atom3, atom4) in enumerate(angles):
               for atom, suffix in zip([atom1, atom2, atom3, atom4], ["A1", "A2", "A3", "A4"]):
                     label = f"{angle_type}_{idx}:{suffix}"
                     if atom not in atom_labels:
                        atom_labels[atom] = []
                     atom_labels[atom].append(label)

         # Set combined labels for each atom
         for atom, labels in atom_labels.items():
            mol.GetAtomWithIdx(atom).SetProp("atomNote", ", ".join(labels))

         # Draw molecule
         return Draw.MolToImage(mol, size=(800, 800))

