"""Learn a potential energy surface with the MBTR representation

MBTR is an easy route for learning a forcefield because it represents
a molecule as a single vector, which means we can case the learning
problem as a simple "molecule->energy" learning problem. Other methods,
such as SOAP, provided atomic-level features that must require an
extra step "molecule->atoms->energy/atom->energy".
"""
from shutil import rmtree

from ase.calculators.calculator import Calculator, all_changes
from ase.vibrations import Vibrations
from ase import Atoms
from sklearn.linear_model import LinearRegression
from dscribe.descriptors import MBTR
import numpy as np

from jitterbug.model.base import EnergyModel


class MBTRCalculator(Calculator):
    """A learnable forcefield based on GPR and fingerprints computed using DScribe"""

    implemented_properties = ['energy', 'forces']
    default_parameters = {
        'descriptor': MBTR(
            species=["H", "C", "N", "O"],
            geometry={"function": "inverse_distance"},
            grid={"min": 0, "max": 1, "n": 100, "sigma": 0.1},
            weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
            periodic=False,
            normalization="l2",
        ),
        'model': LinearRegression(),
        'intercept': 0.,  # Normalizing parameters
        'scale': 0.
    }

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        # Compute the energy using the learned model
        desc = self.parameters['descriptor'].create_single(atoms)
        energy_no_int = self.parameters['model'].predict(desc[None, :])
        self.results['energy'] = energy_no_int[0] * self.parameters['scale'] + self.parameters['intercept']

        # If desired, compute forces numerically
        if 'forces' in properties:
            # calculate_numerical_forces use that the calculation of the input atoms,
            #  even though it is a method of a calculator and not of the input atoms :shrug:
            temp_atoms: Atoms = atoms.copy()
            temp_atoms.calc = self
            self.results['forces'] = self.calculate_numerical_forces(temp_atoms)

    def train(self, train_set: list[Atoms]):
        """Train the embedded forcefield object

        Args:
            train_set: List of Atoms objects containing at least the energy
        """

        # Determine the mean energy and subtract it off
        energies = np.array([atoms.get_potential_energy() for atoms in train_set])
        self.parameters['intercept'] = energies.mean()
        energies -= self.parameters['intercept']
        self.parameters['scale'] = energies.std()
        energies /= self.parameters['scale']

        # Compute the descriptors and use them to fit the model
        desc = self.parameters['descriptor'].create(train_set)
        self.parameters['model'].fit(desc, energies)


class MBTREnergyModel(EnergyModel):
    """Use the MBTR representation to model the potential energy surface

    Args:
        calc: Calculator used to fit the potential energy surface
        reference: Reference structure at which we compute the Hessian
    """

    def __init__(self, calc: MBTRCalculator, reference: Atoms):
        super().__init__()
        self.calc = calc
        self.reference = reference

    def train(self, data: list[Atoms]) -> MBTRCalculator:
        self.calc.train(data)
        return self.calc

    def mean_hessian(self, model: MBTRCalculator) -> np.ndarray:
        self.reference.calc = model
        try:
            vib = Vibrations(self.reference, name='mbtr-temp')
            vib.run()
            return vib.get_vibrations().get_hessian_2d()
        finally:
            rmtree('mbtr-temp', ignore_errors=True)
