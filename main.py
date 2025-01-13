from experiment import Experiment

# TODO: fix half precision

config = {
      "general" : {
         "device": "cuda",
         "float_precision": 32,
         "path_to_data": "/home/sebidom/dom/manifold_contgfn/manifold-gfn/smiles_strings.npy",
         "T" : 298.15,
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
         "uniform": True,
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
      "logging" : {
          "log_directory": "/home/sebidom/dom/manifold_contgfn/manifold-gfn/logs",
          "log_interval": 50,
          "checkpoint_interval": 200,
          "checkpoints": True,
          "n_uniform_samples": 10000,
          "n_onpolicy_samples": 10000,
          "num_bins": 25,
      }
}

# exp = Experiment(checkpoint="/home/sebidom/dom/manifold_contgfn/manifold-gfn/logs/2025-01-13-17-19-08/checkpoint.pt")
exp = Experiment(config=config)
exp.train(100000)






