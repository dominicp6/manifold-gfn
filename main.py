import wandb

from process_data import SMILESDataset
from conformer_environment import ConformerEnvironment
from proxy import TorchANIMoleculeEnergy
from policy import Policy
from gflownet import GFlowNet

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)  # Recursively turn nested dicts into Config
            setattr(self, key, value)

config = {
      "general" : {
         "device": "cpu",
         "float_precision": 32,
      },
      "proxy" : {
         "model": "ANI2x",
         "skip_setup": True,
         "max_energy": -2495.3022,
         "min_energy": -2501.4858,
      },
      "env" : {
         "single_molecule": True,
         "molecule_id": 1,
         "remove_hydrogens": True,
         "n_comp": 3,
         "traj_length": 5,
         "encoding_multiplier": 5,
         "vonmises_min_concentration": 1e-3,
      },
      "forward_policy" : {
         "n_hid": 256,
         "n_layers": 3,
      },
      "backward_policy" : {
         "n_hid": 256,
         "n_layers": 3,
      },
      "optimizer_config" : {
         "loss": "trajectorybalance",
         "lr": 0.0001,
         "lr_z_mult": 10,
         "lr_decay_period": 1000000,
         "lr_decay_gamma": 0.5,
         "z_dim": 16,
         "initial_z_scaling": 50.0,
         "method": "adam",
         "early_stopping": 0.0,
         "adam_beta1": 0.9,
         "adam_beta2": 0.999,
         "clip_value": 1e-7,
         "steps_per_batch": 3,
         "gradient_clipping": True,
      },
      "gflownet" : {
         "regular_capacity": 1000,
         "priority_capacity": 500,
         "priority_ratio": 0.5,
         "batch_size": {
            "forward": 16,
            "replay": 16,
         },
      },
}

wandb.init(project="debug_run", config=config)
config = Config(wandb.config)

device = config.general.device
float_precision = config.general.float_precision

# Load data
path_to_data = "/home/sebidom/dom/manifold_contgfn/manifold-gfn/smiles_strings.npy"
dataset = SMILESDataset(path_to_data)

# Init proxy
proxy = TorchANIMoleculeEnergy(model = config.proxy.model, device=device, float_precision=float_precision, skip_setup=config.proxy.skip_setup, max_energy=-config.proxy.max_energy, min_energy=config.proxy.min_energy)

# Init environment
env = ConformerEnvironment(dataset=dataset,
                           single_molecule=config.env.single_molecule,
                           molecule_id=config.env.molecule_id,
                           remove_hydrogens=config.env.remove_hydrogens,
                           n_comp=config.env.n_comp,
                           traj_length=config.env.traj_length,
                           encoding_multiplier=config.env.encoding_multiplier,
                           vonmises_min_concentration=config.env.vonmises_min_concentration,
                           proxy=proxy,
                           device=device)

# Init policy networks
# TODO: both MLPs for now, might consider uniform backward policy
forward_policy = Policy(env=env, 
                        device=device, 
                        float_precision=float_precision, 
                        n_hid=config.forward_policy.n_hid,
                        n_layers=config.forward_policy.n_layers)

backward_policy = Policy(env=env, 
                         device=device, 
                         float_precision=float_precision, 
                         n_hid=config.backward_policy.n_hid, 
                         n_layers=config.backward_policy.n_layers)

# Init GFlowNet
gfn = GFlowNet(env = env,
               device = device,
               forward_policy=forward_policy,
               backward_policy=backward_policy,
               optimizer_config=config.optimizer_config, 
               batch_size={"forward": 16, "replay": 16})

# Train
gfn.train(n_train_steps=100000)






