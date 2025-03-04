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

    def __init__(self, device, float_precision):
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)

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


class MoleculeEnergyBase(Proxy, ABC):
    def __init__(self, device, float_precision, config):
        super().__init__(device=device, float_precision=float_precision)

        if config.remove_outliers and not config.clamp:
            warnings.warn("If outliers are removed it's recommended to also clamp the values.")

        """
        Parameters
        ----------

        batch_size : int
            Batch size for the underlying model.

        n_samples : int
            Number of samples that will be used to estimate minimum and maximum energy.

        normalise : bool
            Whether to truncate the energies to a (0, 1) range (estimated based on
            sample conformers).

        remove_outliers : bool
            Whether to adjust the min and max energy values estimated on the sample of
            conformers by removing 0.01 quantiles.

        clamp : bool
            Whether to clamp the energies to the estimated min and max values.
        """
        self.n_samples = config.n_samples
        self.normalise = config.normalise
        self.remove_outliers = config.remove_outliers
        self.clamp = config.clamp
        self.max_energy = config.max_energy
        self.min_energy = config.min_energy
        self.skip_setup = config.skip_setup
        self.max_batch_size = config.max_batch_size

    @abstractmethod
    def compute_energies(self, states: Tensor, env) -> Tensor:
        pass

    def __call__(self, atomic_numbers, conformations_positions) -> Tensor:
        energies = self.compute_energies(atomic_numbers, conformations_positions)

        if self.clamp:
            energies = energies.clamp(self.min_energy, self.max_energy)

        energies = energies - self.min_energy
        # energies = energies - self.max_energy

        # if self.normalise:
            # energies = energies / (self.max_energy - self.min_energy)

        return energies

    def setup(self, env):
        if self.skip_setup:
            assert self.max_energy is not None and self.min_energy is not None, "If skip_setup is True, max_energy and min_energy must be provided."
        else:
            randomly_sampled_states = 2 * np.pi * np.random.rand(self.n_samples, env.n_dim)
            energies = self.compute_energies(*env.statebatch2conformerbatch(randomly_sampled_states)).cpu().numpy()

            self.max_energy = max(energies)
            self.min_energy = min(energies)

            if self.remove_outliers:
                self.max_energy = np.quantile(energies, 0.99)
                self.min_energy = np.quantile(energies, 0.01)
            
            print(f"Estimated min energy: {self.min_energy}")
            print(f"Estimated max energy: {self.max_energy}")


TORCHANI_MODELS = {
    "ANI1x": torchani.models.ANI1x,
    "ANI1ccx": torchani.models.ANI1ccx,
    "ANI2x": torchani.models.ANI2x,
}


class TorchANIMoleculeEnergy(MoleculeEnergyBase):
    def __init__(
        self,
        device,
        float_precision,
        config,
    ):
        """
        Parameters
        ----------
        model : str
            The name of the pretrained model to be used for prediction.

        normalize : bool
            Whether to truncate the energies to a (0, 1) range (estimated based on
            sample conformers).
        """
        super().__init__(device=device, float_precision=float_precision, config=config)

        # TODO: use float precision correctly here

        if TORCHANI_MODELS.get(config.model) is None:
            raise ValueError(
                f'Tried to use model "{config.model}", '
                f"but only {set(TORCHANI_MODELS.keys())} are available."
            )

        self.model = TORCHANI_MODELS[config.model](periodic_table_index=True, model_index=None).to(self.device)

    @torch.no_grad()
    def compute_energies(self, atomic_numbers, conformations_positions) -> torch.Tensor:
        """
        Compute the energies of a batch of molecular conformations using the ANI2X model.

        Args:
            atomic_numbers (torch.Tensor): 2D tensor containing atomic numbers for the molecule (shape: [batch_size, N_atoms]).
            conformations_positions (torch.Tensor): 3D tensor of shape [batch_size, N_atoms, 3] containing xyz positions for each conformation.

        Returns:
            torch.Tensor: Energies of the batch of molecular conformations as a 1D tensor (shape: [batch_size]).
        """
        # Validate input shapes
        assert conformations_positions.ndim == 3, "conformations_positions must be a 3D tensor [batch_size, N_atoms, 3]"
        assert atomic_numbers.ndim == 2, "atomic_numbers must be a 2D tensor [batch_size, N_atoms]"
        assert atomic_numbers.shape[:2] == conformations_positions.shape[:2], \
            "Mismatch between number of atoms in atomic_numbers and conformations_positions"

        batch_size = conformations_positions.shape[0]
        energies = torch.empty(batch_size, dtype=torch.float32)

        if batch_size > self.max_batch_size:
            # Process in blocks
            for start in range(0, batch_size, self.max_batch_size):
                end = min(start + self.max_batch_size, batch_size)
                block_atomic_numbers = atomic_numbers[start:end]
                block_positions = conformations_positions[start:end]
                energies[start:end] = self.model((block_atomic_numbers, block_positions)).energies.float()
        else:
            # Single batch
            energies = self.model((atomic_numbers, conformations_positions)).energies.float()

        return energies

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        new_obj.n_samples = self.n_samples
        new_obj.max_energy = self.max_energy
        new_obj.min = self.min
        new_obj.model = self.model
        return new_obj
