"""Models which use a single vector to describe a molecule.

Such approaches, such as MBTR, present a simple "molecule->energy" learning problem.
Other methods, such as SOAP, provided atomic-level features that must require an
extra step "molecule->atoms->energy/atom->energy".

We train the models using sklearn for this example.
"""
from typing import List

from dscribe.descriptors.descriptorglobal import DescriptorGlobal
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, clone
from dscribe.descriptors import MBTR
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
import numpy as np

from jitterbug.model.base import ASEEnergyModel


class DScribeGlobalCalculator(Calculator):
    """A learnable forcefield based on global fingerprints computed using DScribe"""

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
        'offset': 0.,  # Normalizing parameters
        'scale': 1.
    }

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        # Compute the energy using the learned model
        desc = self.parameters['descriptor'].create_single(atoms)
        energy_unscaled = self.parameters['model'].predict(desc[None, :])
        self.results['energy'] = energy_unscaled[0] * self.parameters['scale'] + self.parameters['offset']

        # If desired, compute forces numerically
        if 'forces' in properties:
            # calculate_numerical_forces use that the calculation of the input atoms,
            #  even though it is a method of a calculator and not of the input atoms :shrug:
            temp_atoms: Atoms = atoms.copy()
            temp_atoms.calc = self
            self.results['forces'] = self.calculate_numerical_forces(temp_atoms)


class DScribeGlobalEnergyModel(ASEEnergyModel):
    """Compute energy using a scikit-learn model that computes energies from global descriptors

    Args:
        reference: Reference structure at which we compute the Hessian
        descriptors: Tool used to compute descriptors
        model: Scikit-learn model to use to compute energies given descriptors
        num_calculators: Number of models to use in ensemble
    """

    def __init__(self, reference: Atoms, descriptors: DescriptorGlobal, model: BaseEstimator, num_calculators: int):
        super().__init__(reference, num_calculators)
        self.desc = descriptors
        self.model = model

    def train_calculator(self, data: List[Atoms]) -> Calculator:
        # Make a copy of the model
        model: BaseEstimator = clone(self.model)

        # Compute the descriptors then train
        train_x = self.desc.create(data)
        train_y = np.array([a.get_potential_energy() for a in data])
        scale_y, offset_y = np.std(train_y), np.mean(train_y)
        train_y = (train_y - offset_y) / scale_y
        model.fit(train_x, train_y)

        # Assemble into a Calculator
        return DScribeGlobalCalculator(
            descriptor=self.desc,
            model=model,
            offset=offset_y,
            scale=scale_y,
        )
