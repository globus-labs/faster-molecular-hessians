"""Models which treat each term in the Hessian as independent

We achieve a approximate Hessian by conditioning a linear model
on the assumption that the Hessian contains parameters which are zero
and that smaller parameters are more likely than larger
"""
import numpy as np
from ase import Atoms
from sklearn.linear_model import ARDRegression
from sklearn.linear_model._base import LinearModel

from .base import EnergyModel


def get_displacement_matrix(atoms: Atoms, reference: Atoms) -> np.ndarray:
    """Get the displacements of a structure from a reference
    in the order, as used in a Hessian calculation.

    Args:
        atoms: Displaced structure
        reference: Reference structure
    Returns:
        Vector of displacements
    """

    # Compute the displacements
    disp_matrix = (reference.positions - atoms.positions).flatten()
    disp_matrix = disp_matrix[:, None] * disp_matrix[None, :]

    # Multiply the off-axis terms by two, as they appear twice in the energy model
    n_terms = len(atoms) * 3
    off_diag = np.triu_indices(n_terms, k=1)
    disp_matrix[off_diag] *= 2

    # Return the upper triangular matrix
    return disp_matrix[np.triu_indices(n_terms)]


class LinearHessianModel(EnergyModel):
    """Fits a model for energy using linear regression

    Implicitly treats all elements of the Hessian matrix as unrelated

    Args:
        reference: Fully-relaxed structure used as the reference
        regressor: LinearRegression class used to learn model
    """

    def __init__(self, reference: Atoms, regressor: type[LinearModel] = ARDRegression):
        self.reference = reference
        self.regressor = regressor

    def train(self, data: list[Atoms]) -> ARDRegression:
        # X: Displacement vectors for each
        x = [get_displacement_matrix(atoms, self.reference) for atoms in data]
        x = np.multiply(x, 0.5)

        # Y: Subtract off the reference energy
        ref_energy = self.reference.get_potential_energy()
        y = [atoms.get_potential_energy() - ref_energy for atoms in data]

        # Fit the ARD model and ensure it captures the data well
        model = self.regressor(fit_intercept=False).fit(x, y)
        pred = model.predict(x)
        max_error = np.abs(pred - y).max()
        if max_error > 0.001:
            raise ValueError(f'Model error exceeds 1 meV. Actual: {max_error:.2e}')

        return model

    def mean_hessian(self, model: ARDRegression) -> np.ndarray:
        return self._params_to_hessian(model.coef_)

    def sample_hessians(self, model: ARDRegression, num_samples: int) -> list[np.ndarray]:
        # Sample the model parameters
        params = np.random.multivariate_normal(model.coef_, model.sigma_, size=num_samples)

        # Assemble them into Hessians
        output = []
        for param in params:
            hessian = self._params_to_hessian(param)
            output.append(hessian)
        return output

    def _params_to_hessian(self, param: np.ndarray) -> np.ndarray:
        """Convert the parameters for the linear model into a Hessian"""
        # Get the parameters
        n_terms = len(self.reference) * 3
        triu_inds = np.triu_indices(n_terms)
        off_diag_triu_inds = np.triu_indices(n_terms, k=1)

        # Assemble the hessian
        hessian = np.zeros((n_terms, n_terms))
        hessian[triu_inds] = param
        hessian[off_diag_triu_inds] /= 2
        hessian.T[triu_inds] = hessian[triu_inds]
        return hessian
