"""Simple, friendly, random sampling"""
from dataclasses import dataclass

import numpy as np
import ase

from jitterbug.sampler.base import StructureSampler


@dataclass
class UniformSampler(StructureSampler):
    """Sample randomly-chosen directions

    Perturbs each atom in each direction a random amount between -:attr:`step_size` and :attr:`step_size`.
    """

    step_size: float = 0.005
    """Amount to displace the atoms (units: Angstrom)"""

    @property
    def name(self) -> str:
        return f'uniform_{self.step_size:.3e}'

    def produce_structures(self, atoms: ase.Atoms, count: int, seed: int = 1) -> list[ase.Atoms]:
        # Make the RNG
        n_atoms = len(atoms)
        rng = np.random.RandomState(seed + n_atoms)

        output = []
        for _ in range(count):
            # Sample a perturbation
            disp = rng.normal(-self.step_size, self.step_size, size=(n_atoms, 3))

            # Make the new atoms
            new_atoms = atoms.copy()
            new_atoms.positions += disp
            output.append(new_atoms)
        return output
