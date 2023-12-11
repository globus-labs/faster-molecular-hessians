from pathlib import Path

import ase
import numpy as np
from colmena.queue import ColmenaQueues
from colmena.thinker import BaseThinker, ResourceCounter


class HessianThinker(BaseThinker):
    """Base class for thinkers

    Implementations must write their simulation data to the same spot"""

    atoms: ase.Atoms
    """Unperturbed atomic structure"""

    run_dir: Path
    """Path to the run directory"""
    result_file: Path
    """Path to file in which to store result records"""

    def __init__(self, queues: ColmenaQueues, rec: ResourceCounter, run_dir: Path, atoms: ase.Atoms):
        super().__init__(queues, rec)
        self.atoms = atoms

        # Prepare for outputs
        self.run_dir = run_dir
        self.run_dir.mkdir(exist_ok=True)
        self.result_file = run_dir / 'simulation-results.json'

    def compute_hessian(self) -> np.ndarray:
        """Compute the Hessian using finite differences

        Returns:
            Hessian in the 2D form
        Raises:
            (ValueError) If there is missing data
        """
        raise NotImplementedError()
