import os
import random
from os.path import expandvars
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import get_original_cwd, instantiate
from omegaconf import OmegaConf
from torchtyping import TensorType


def set_device(device: Union[str, torch.device]):
    if isinstance(device, torch.device):
        return device
    if device.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_float_precision(precision: Union[int, torch.dtype]):
    if isinstance(precision, torch.dtype):
        return precision
    if precision == 16:
        return torch.float16
    elif precision == 32:
        return torch.float32
    elif precision == 64:
        return torch.float64
    else:
        raise ValueError("Precision must be one of [16, 32, 64]")


def set_int_precision(precision: Union[int, torch.dtype]):
    if isinstance(precision, torch.dtype):
        return precision
    if precision == 16:
        return torch.int16
    elif precision == 32:
        return torch.int32
    elif precision == 64:
        return torch.int64
    else:
        raise ValueError("Precision must be one of [16, 32, 64]")


def torch2np(x):
    if hasattr(x, "is_cuda") and x.is_cuda:
        x = x.detach().cpu()
    return np.array(x)


def find_latest_checkpoint(ckpt_dir, pattern):
    final = list(ckpt_dir.glob(f"{pattern}*final*"))
    if len(final) > 0:
        return final[0]
    ckpts = list(ckpt_dir.glob(f"{pattern}*"))
    if not ckpts:
        raise ValueError(f"No checkpoints found in {ckpt_dir} with pattern {pattern}")
    return sorted(ckpts, key=lambda f: float(f.stem.split("iter")[1]))[-1]


def tfloat(x, device, float_type):
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).type(float_type).to(device)
    if torch.is_tensor(x):
        return x.type(float_type).to(device)
    else:
        return torch.tensor(x, dtype=float_type, device=device)


def tlong(x, device):
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).type(torch.long).to(device)
    if torch.is_tensor(x):
        return x.type(torch.long).to(device)
    else:
        return torch.tensor(x, dtype=torch.long, device=device)


def tint(x, device, int_type):
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).type(int_type).to(device)
    if torch.is_tensor(x):
        return x.type(int_type).to(device)
    else:
        return torch.tensor(x, dtype=int_type, device=device)


def tbool(x, device):
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).type(torch.bool).to(device)
    if torch.is_tensor(x):
        return x.type(torch.bool).to(device)
    else:
        return torch.tensor(x, dtype=torch.bool, device=device)


def concat_items(list_of_items, index=None):
    if isinstance(list_of_items[0], np.ndarray):
        result = np.concatenate(list_of_items)
        if index is not None:
            index = index.cpu().numpy()
            result = result[index]
    elif torch.is_tensor(list_of_items[0]):
        result = torch.cat(list_of_items)
        if index is not None:
            result = result[index]
    else:
        raise NotImplementedError(
            "cannot concatenate {}".format(type(list_of_items[0]))
        )

    return result


def extend(
    orig: Union[List, TensorType["..."]], new: Union[List, TensorType["..."]]
) -> Union[List, TensorType["..."]]:
    assert type(orig) == type(new)
    if isinstance(orig, list):
        orig.extend(new)
    elif torch.tensor(orig):
        orig = torch.cat([orig, new])
    else:
        raise NotImplementedError(
            "Extension only supported for lists and torch tensors"
        )
    return orig


def copy(x: Union[List, TensorType["..."]]):
    if torch.is_tensor(x):
        return x.clone().detach()
    else:
        return x.copy()


def chdir_random_subdir():
    """
    Creates a directory with random name and changes current working directory to it.

    Aimed as a hotfix for race conditions: currently, by default, the directory in
    which the experiment will be logged is named based on the current timestamp. If
    multiple jobs start at exactly the same time, they can be trying to log to
    the same directory. In particular, this causes issues when using dataset
    evaluation (e.g., JSD computation).
    """
    cwd = os.getcwd()
    cwd += "/%08x" % random.getrandbits(32)
    os.mkdir(cwd)
    os.chdir(cwd)
