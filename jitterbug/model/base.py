"""Base class defining the key methods"""
import numpy as np
from ase import Atoms


class EnergyModel:
    """Base class for functions which predict energy given molecular structure"""

    def train(self, data: list[Atoms]) -> object:
        """Produce an energy model given observations of energies

        Args:
            data: Energy evaluations
        Returns:
            Model trained to predict energy given atomic structure
        """
        raise NotImplementedError()

    def mean_hessian(self, model: object) -> np.ndarray:
        """Produce the most-likely Hessian given the model

        Args:
            model: Model trained by this class
        Returns:
            The most-likely Hessian given the model
        """

    def sample_hessians(self, model: object, num_samples: int) -> list[np.ndarray]:
        """Produce estimates for the Hessian given the model

        Args:
            model: Model trained by this class
            num_samples: Number of Hessians to sample
        Returns:
            A list of 2D hessians
        """
        raise NotImplementedError()
