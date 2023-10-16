"""Base class defining the key methods"""
from tempfile import TemporaryDirectory
from typing import List, Optional
from random import choices
from pathlib import Path
import os

import numpy as np
from scipy import stats
from ase import Atoms
from ase.vibrations import Vibrations
from ase.calculators.calculator import Calculator


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


class ASEEnergyModel(EnergyModel):
    """Energy models which produce a series of ASE :class:`~ase.calculators.calculator.Calculator` objects"""

    def __init__(self, reference: Atoms, num_calculators: int, scratch_dir: Optional[Path] = None):
        self.reference = reference
        self.scratch_dir = scratch_dir
        self.num_calculators = num_calculators

    def train_calculator(self, data: List[Atoms]) -> Calculator:
        """Train one of the constituent calculators

        Args:
            data: Training data
        Returns:
            Retrained calculator
        """
        raise NotImplementedError()

    def train(self, data: list[Atoms]) -> List[Calculator]:
        # Train each on a different bootstrap sample
        output = []
        for _ in range(self.num_calculators):
            sampled_data = choices(data, k=len(data))
            output.append(self.train_calculator(sampled_data))
        return output

    def compute_hessian(self, mol: Atoms, calc: Calculator) -> np.ndarray:
        """Compute the Hessian using one of the calculators

        Args:
            mol: Molecule to be evaluated
            calc: Calculator to use
        Returns:
            2D Hessian matrix
        """
        # Clone the molecule to avoid any cross-talk
        mol = mol.copy()

        with TemporaryDirectory(dir=self.scratch_dir) as td:
            start_dir = Path.cwd()
            try:
                # Run the vibrations calculation in a temporary directory
                os.chdir(td)
                mol.calc = calc
                vib = Vibrations(mol, name='mbtr-temp')
                vib.run()

                # Return only the 2D Hessian
                return vib.get_vibrations().get_hessian_2d()
            finally:
                os.chdir(start_dir)

    def mean_hessian(self, models: list[Calculator]) -> np.ndarray:
        # Run all calculators
        hessians = [self.compute_hessian(self.reference, calc) for calc in models]

        # Return the mean
        return np.mean(hessians, axis=0)

    def sample_hessians(self, models: list[Calculator], num_samples: int) -> list[np.ndarray]:
        # Run all calculators
        hessians = [self.compute_hessian(self.reference, calc) for calc in models]

        # Compute the mean and covariance for each parameter
        triu_ind = np.triu_indices(hessians[0].shape[0])
        hessians_flat = [h[triu_ind] for h in hessians]
        hessian_mean = np.mean(hessians_flat, axis=0)
        hessian_covar = np.cov(hessians_flat, rowvar=False)

        # Generate samples
        hessian_mvn = stats.multivariate_normal(hessian_mean, hessian_covar, allow_singular=True)
        diag_indices = np.diag_indices(hessians[0].shape[0])
        output = []
        for sample in hessian_mvn.rvs(size=num_samples):
            # Fill in a 2D version
            sample_hessian = np.zeros_like(hessians[0])
            sample_hessian[triu_ind] = sample

            # Make it symmetric
            sample_hessian += sample_hessian.T
            sample_hessian[diag_indices] /= 2

            output.append(sample_hessian)
        return output
