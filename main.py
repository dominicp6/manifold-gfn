from process_data import SMILESDataset
from conformer_environment import ConformerEnvironment
from proxy import TorchANIMoleculeEnergy
from policy import Policy
from gflownet import GFlowNet
from optimizer import OptimizerConfig

device = 'cpu'
float_precision = 32

# Load data
path_to_data = "/home/sebidom/dom/manifold_contgfn/manifold-gfn/smiles_strings.npy"
dataset = SMILESDataset(path_to_data)

# Init proxy
proxy = TorchANIMoleculeEnergy(model = "ANI2x", device=device, float_precision=float_precision, skip_setup=True)

# Init environment
env = ConformerEnvironment(dataset=dataset,
                           single_molecule=True,
                           molecule_id=1,
                           remove_hydrogens=True,
                           n_comp=3,
                           traj_length=5,
                           encoding_multiplier=5,
                           vonmises_min_concentration=1e-3,
                           proxy=proxy,
                           device=device)

# Init policy networks
# TODO: both MLPs for now, might consider uniform backward policy
forward_policy = Policy(env=env, 
                        device=device, 
                        float_precision=float_precision, 
                        n_hid=256, 
                        n_layers=3)
backward_policy = Policy(env=env, 
                         device=device, 
                         float_precision=float_precision, 
                         n_hid=256, 
                         n_layers=3)
optimizer_config = OptimizerConfig()

# Init GFlowNet
gfn = GFlowNet(env = env,
               device = device,
               forward_policy=forward_policy,
               backward_policy=backward_policy,
               optimizer_config=optimizer_config, 
               batch_size={"forward": 16, "replay": 16},
            #    clip_value=1e-7,
               replay_sampling=False)

# Train
gfn.train(n_train_steps=1000)






