"""
Runnable script with hydra capabilities
"""

import time 
import logging

# This is a hotfix for tblite (used for the conformer generation) not
# importing correctly unless it is being imported first.
try:
    from tblite import interface
except:
    pass

import os
import pickle
import random
import sys

import hydra
import pandas as pd

from gflownet.utils.common import chdir_random_subdir
from gflownet.utils.policy import parse_policy_config

time_logger = logging.getLogger("initialization_time")
time_logger.setLevel(logging.INFO)

@hydra.main(config_path="./config", config_name="main", version_base="1.1")
def main(config):
    # TODO: fix race condition in a more elegant way
    chdir_random_subdir()

    # Get current directory and set it as root log dir for Logger
    cwd = os.getcwd()
    config.logger.logdir.root = cwd
    print(f"\nLogging directory of this run:  {cwd}\n")

    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)

    # Logger
    start_time = time.time()
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)
    time_logger.info(f"Logger initialization time: {time.time() - start_time:.4f}s")
    # The proxy is required in the env for scoring: might be an oracle or a model
    start_time = time.time()
    proxy = hydra.utils.instantiate(
        config.proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    time_logger.info(f"Proxy initialization time: {time.time() - start_time:.4f}s")
    # The proxy is passed to env and used for computing rewards
    start_time = time.time()
    env = hydra.utils.instantiate(
        config.env,
        proxy=proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    time_logger.info(f"Environment initialization time: {time.time() - start_time:.4f}s")
    # The policy is used to model the probability of a forward/backward action
    start_time = time.time()
    forward_config = parse_policy_config(config, kind="forward")
    backward_config = parse_policy_config(config, kind="backward")

    forward_policy = hydra.utils.instantiate(
        forward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
    )
    backward_policy = hydra.utils.instantiate(
        backward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
        base=forward_policy,
    )
    time_logger.info(f"Policies initialization time: {time.time() - start_time:.4f}s")

    start_time = time.time()
    gflownet = hydra.utils.instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env=env,
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        buffer=config.env.buffer,
        logger=logger,
    )
    time_logger.info(f"GFlowNet initialization time: {time.time() - start_time:.4f}s")
    gflownet.train()

    # Sample from trained GFlowNet
    if config.n_samples > 0 and config.n_samples <= 1e5:
        batch, times = gflownet.sample_batch(n_forward=config.n_samples, train=False)
        x_sampled = batch.get_terminating_states(proxy=True)
        energies = env.oracle(x_sampled)
        x_sampled = batch.get_terminating_states()
        df = pd.DataFrame(
            {
                "readable": [env.state2readable(x) for x in x_sampled],
                "energies": energies.tolist(),
            }
        )
        df.to_csv("gfn_samples.csv")
        dct = {"x": x_sampled, "energy": energies}
        pickle.dump(dct, open("gfn_samples.pkl", "wb"))
        # TODO: refactor before merging
        dct["conformer"] = [env.set_conformer(state).rdk_mol for state in x_sampled]
        pickle.dump(
            dct, open(f"conformers_{env.smiles}_{type(env.proxy).__name__}.pkl", "wb")
        )

    # Print replay buffer
    if len(gflownet.buffer.replay) > 0:
        print("\nReplay buffer:")
        print(gflownet.buffer.replay)

    # Close logger
    gflownet.logger.end()


def set_seeds(seed):
    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
    sys.exit()
