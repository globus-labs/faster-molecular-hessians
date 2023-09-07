"""Learn a potential energy surface with the MBTR representation

MBTR is an easy route for learning a forcefield because it represents
a molecule as a single vector, which means we can case the learning
problem as a simple "molecule->energy" learning problem. Other methods,
such as SOAP, provided atomic-level features that must require an
extra step "molecule->atoms->energy/atom->energy".
"""

from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
from sklearn.linear_model import LinearRegression
from dscribe.descriptors import MBTR
import numpy as np


class MBTRCalculator(Calculator):
    """A learnable forcefield based on GPR and fingerprints computed using DScribe"""

    implemented_properties = ['energy', 'forces']
    default_parameters = {
        'descriptor': MBTR(
            species=["C", "H", "O"],
            geometry={"function": "inverse_distance"},
            grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1},
            weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
            periodic=False,
            normalization="l2",
        ),
        'model': LinearRegression(),
        'intercept': 0.
    }

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        # Compute the energy using the learned model
        desc = self.parameters['descriptor'].create_single(atoms)
        energy_no_int = self.parameters['model'].predict(desc[None, :])
        self.results['energy'] = energy_no_int[0] + self.parameters['intercept']

    def train(self, train_set: list[Atoms]):
        """Train the embedded forcefield object

        Args:
            train_set: List of Atoms objects containing at least the energy
        """

        # Determine the mean energy and subtract it off
        energies = np.array([atoms.get_potential_energy() for atoms in train_set])
        self.parameters['intercept'] = energies.mean()
        energies -= self.parameters['intercept']

        # Compute the descriptors and use them to fit the model
        desc = self.parameters['descriptor'].create(train_set)
        self.parameters['model'].fit(desc, energies)
