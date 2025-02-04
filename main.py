from experiment import Experiment

config = {
      "general" : {
         "device": "cuda",
         "float_precision": 32,
         "path_to_data": "/home/sebidom/dom/manifold_contgfn/manifold-gfn/dipeptides_smiles.hdf5",
         "T" : 1000,
      },
      "proxy" : {
         "model": "ANI2x",
         "skip_setup": False,
         "normalise": True, 
         "clamp": True,
         "remove_outliers": True,
         "n_samples": 10000,
         "max_energy": -2495.3022,
         "min_energy": -2501.4858,
         "max_batch_size": 2500,
      },
      "env" : {
         "single_molecule": True,
         "molecule_id": 1,
         "remove_hydrogens": True,
         "n_comp": 5,
         "traj_length": 5,
         "encoding_multiplier": 5,
         "vonmises_max_log_conc": 7,
         "vonmises_min_log_conc": 1,
         "backbone_only": True,
      },
      "forward_policy" : {
         "n_hid": 512,
         "n_layers": 5,
      },
      "backward_policy" : {
         "uniform": False,
         "n_hid": 512,
         "n_layers": 5,
      },
      "optimizer_config" : {
         "loss": "trajectorybalance",
         "lr": 0.0001,
         "lr_z_mult": 10,
         "lr_decay_period": 5000,
         "lr_decay_gamma": 0.9,
         "z_dim": 16,
         "initial_z_scaling": 25.0,
         "method": "adam",
         "early_stopping": 0.0,
         "adam_beta1": 0.9,
         "adam_beta2": 0.999,
         "clip_value": 1e-4,
         "steps_per_batch": 3,
         "gradient_clipping": True,
      },
      "gflownet" : {
         "regular_capacity": 1000,
         "priority_capacity": 1000,
         "priority_ratio": 0.8,
         "log_reward_min": -30,
         "batch_size": {
            "forward": 80,
            "replay": 20,
         },
      },
      "logging" : {
          "log_directory": "/home/sebidom/dom/manifold_contgfn/manifold-gfn/logs",
          "log_interval": 200,
          "checkpoint_interval": 10000,
          "visualisation_interval": 1000, 
          "checkpoints": True,
          "n_uniform_samples": 100000,
          "n_onpolicy_samples": 20000,
          "num_bins": 20,
      }
}
# exp = Experiment(checkpoint="/home/sebidom/dom/manifold_contgfn/manifold-gfn/logs/2025-01-13-17-19-08/checkpoint.pt")
exp = Experiment(config=config)
exp.train(100000)






