import warnings
from typing import Optional
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import torch
import torchani
from torch import Tensor

from utils.common import set_device, set_float_precision

class Proxy(ABC):
    """
    Generic proxy class
    """

    def __init__(self, device, float_precision, higher_is_better=False, **kwargs):
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Reward2Proxy multiplicative factor (1 or -1)
        self.higher_is_better = higher_is_better

    def setup(self, env):
        pass

    @abstractmethod
    def __call__(self, states: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Args:
            states: ndarray
        Function:
            calls the get_reward method of the appropriate Proxy Class (EI, UCB, Proxy,
            Oracle etc)
        """
        pass

    def infer_on_train_set(self):
        """
        Implement this method in specific proxies.
        It should return the ground-truth and proxy values on the proxy's training set.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `infer_on_train_set`."
        )


class MoleculeEnergyBase(Proxy, ABC):
    def __init__(
        self,
        batch_size: Optional[int] = 128,
        # TODO: change back to 10000 after debugging
        n_samples: int = 100,
        normalize: bool = True,
        remove_outliers: bool = True,
        clamp: bool = True,
        skip_setup: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------

        batch_size : int
            Batch size for the underlying model.

        n_samples : int
            Number of samples that will be used to estimate minimum and maximum energy.

        normalize : bool
            Whether to truncate the energies to a (0, 1) range (estimated based on
            sample conformers).

        remove_outliers : bool
            Whether to adjust the min and max energy values estimated on the sample of
            conformers by removing 0.01 quantiles.

        clamp : bool
            Whether to clamp the energies to the estimated min and max values.
        """
        super().__init__(**kwargs)

        if remove_outliers and not clamp:
            warnings.warn(
                "If outliers are removed it's recommended to also clamp the values."
            )

        self.batch_size = batch_size
        self.n_samples = n_samples
        self.normalize = normalize
        self.remove_outliers = remove_outliers
        self.clamp = clamp
        self.max_energy = None
        self.min_energy = None
        self.min = None
        self.skip_setup = skip_setup

    @abstractmethod
    def compute_energies(self, states: Tensor, env) -> Tensor:
        pass

    def __call__(self, atomic_numbers, conformations_positions) -> Tensor:
        energies = self.compute_energies(atomic_numbers, conformations_positions)

        if self.clamp:
            energies = energies.clamp(self.min_energy, self.max_energy)

        energies = energies - self.max_energy

        if self.normalize:
            energies = energies / (self.max_energy - self.min_energy)

        return energies

    def setup(self, env):
        if self.skip_setup:
            self.max_energy = 0
            self.min_energy = -1
            self.min = -1
        else:
            randomly_sampled_states = 2 * np.pi * np.random.rand(self.n_samples, env.n_dim)
            energies = self.compute_energies(*env.statebatch2conformerbatch(randomly_sampled_states)).cpu().numpy()

            self.max_energy = max(energies)
            self.min_energy = min(energies)

            if self.remove_outliers:
                self.max_energy = np.quantile(energies, 0.99)
                self.min_energy = np.quantile(energies, 0.01)

            if self.normalize:
                # TODO: we still haven't made sense of this
                self.min = -1
            else:
                self.min = self.min_energy - self.max_energy


TORCHANI_MODELS = {
    "ANI1x": torchani.models.ANI1x,
    "ANI1ccx": torchani.models.ANI1ccx,
    "ANI2x": torchani.models.ANI2x,
}


class TorchANIMoleculeEnergy(MoleculeEnergyBase):
    def __init__(
        self,
        model: str = "ANI2x",
        use_ensemble: bool = True,
        batch_size: Optional[int] = 128,
        n_samples: int = 10000,
        normalize: bool = True,
        skip_setup: bool = False,
        **kwargs,
    ):
        # TODO: check whether they activated flag use_ensemble in the original code
        """
        Parameters
        ----------
        model : str
            The name of the pretrained model to be used for prediction.

        use_ensemble : bool
            Whether to use whole ensemble of the models for prediction or only the first one.

        batch_size : int
            Batch size for TorchANI. If none, will process all states as a single batch.

        normalize : bool
            Whether to truncate the energies to a (0, 1) range (estimated based on
            sample conformers).
        """
        super().__init__(batch_size=batch_size, n_samples=n_samples, normalize=normalize, skip_setup=skip_setup, **kwargs)

        if TORCHANI_MODELS.get(model) is None:
            raise ValueError(
                f'Tried to use model "{model}", '
                f"but only {set(TORCHANI_MODELS.keys())} are available."
            )

        self.model = TORCHANI_MODELS[model](
            periodic_table_index=True, model_index=None if use_ensemble else 0
        ).to(self.device)

    @torch.no_grad()
    def compute_energies(self, atomic_numbers, conformations_positions) -> Tensor:
        """
        Compute the energies of a batch of molecular conformations using the ANI2X model.

        Args:
            atomic_numbers (torch.Tensor): 1D tensor containing atomic numbers for the molecule (shape: [N_atoms]).
            conformations_positions (torch.Tensor): 3D tensor of shape [batch_size, N_atoms, 3] containing xyz positions for each conformation.
            model (torchani.models.Model): ANI2X model.

        Returns:
            torch.Tensor: Energies of the batch of molecular conformations as a 1D tensor (shape: [batch_size]).
        """
        # Validate input shapes
        assert conformations_positions.ndim == 3, "conformations_positions must be a 3D tensor [batch_size, N_atoms, 3]"
        assert atomic_numbers.ndim == 2, "atomic_numbers must be a 1D tensor [batch_size, N_atoms]"
        assert atomic_numbers.shape[:2] == conformations_positions.shape[:2], \
            "Mismatch between number of atoms in atomic_numbers and conformations_positions"
        
        batch_size = conformations_positions.shape[0]
        energies = torch.empty(batch_size, dtype=torch.float32)
        energies = self.model((atomic_numbers, conformations_positions)).energies.float()  
        
        return energies

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        new_obj.batch_size = self.batch_size
        new_obj.n_samples = self.n_samples
        new_obj.max_energy = self.max_energy
        new_obj.min = self.min
        new_obj.model = self.model
        return new_obj
