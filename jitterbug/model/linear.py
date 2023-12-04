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


def get_model_inputs(atoms: Atoms, reference: Atoms) -> np.ndarray:
    """Get the inputs for the model, which are derived from the displacements
    of the structure with respect to a reference.

    Args:
        atoms: Displaced structure
        reference: Reference structure
    Returns:
        Vector of displacements in the same order as the
    """

    # Compute the displacements and the products of displacement
    disp_matrix = (atoms.positions - reference.positions).flatten()
    disp_prod_matrix = disp_matrix[:, None] * disp_matrix[None, :]

    # Multiply the off-axis terms by two, as they appear twice in the energy model
    n_terms = len(atoms) * 3
    off_diag = np.triu_indices(n_terms, k=1)
    disp_prod_matrix[off_diag] *= 2

    # Append the displacements and products of displacements
    return np.concatenate([
        disp_matrix,
        disp_prod_matrix[np.triu_indices(n_terms)] / 2
    ], axis=0)


class HarmonicModel(EnergyModel):
    """Expresses energy as a Harmonic model (i.e., 2nd degree Taylor series)

    Contains a total of :math:`3N + 3N(3N+1)/2` terms in total, where :math:`N`
    is the number of atoms in the molecule. The first :math:`3N` correspond to the
    linear terms of the model, which are known as the Jacobian matrix, and the
    latter are from the quadratic terms, which are half of the symmetric Hessian matrix.

    Implicitly treats all terms of the model as unrelated, which is the worst case
    for trying to fit the energy of a molecule. However, it is still possible to fit
    the model with a reduced number of terms if we assume that most terms are near zero.

    The energy model is:

    :math:`E = E_0 + \\sum_i J_i \\delta_i + \\frac{1}{2}\\sum_{i,j} H_{i,j}\\delta_i\\delta_j`

    Args:
        reference: Fully-relaxed structure used as the reference
        regressor: LinearRegression class used to learn model
    """

    def __init__(self, reference: Atoms, regressor: type[LinearModel] = ARDRegression):
        self.reference = reference
        self.regressor = regressor

    def train(self, data: list[Atoms]) -> LinearModel:
        # X: Displacement vectors for each
        x = [get_model_inputs(atoms, self.reference) for atoms in data]

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

    def mean_hessian(self, model: LinearModel) -> np.ndarray:
        return self._params_to_hessian(model.coef_)

    def sample_hessians(self, model: LinearModel, num_samples: int) -> list[np.ndarray]:
        # Get the covariance matrix
        if not hasattr(model, 'sigma_'):  # pragma: no-coverage
            raise ValueError(f'Sampling only possible with Bayesian regressors. You trained a {type(model)}')
        if isinstance(model, ARDRegression):
            # The sigma matrix may be zero for high-precision terms
            n_terms = len(model.coef_)
            nonzero_terms = model.lambda_ < model.threshold_lambda

            # Replace those terms (Thanks: https://stackoverflow.com/a/73176327/2593278)
            sigma = np.zeros((n_terms, n_terms))
            sub_sigma = sigma[nonzero_terms, :]
            sub_sigma[:, nonzero_terms] = model.sigma_
            sigma[nonzero_terms, :] = sub_sigma
        else:
            sigma = model.sigma_

        # Sample the model parameters
        params = np.random.multivariate_normal(model.coef_, sigma, size=num_samples)

        # Assemble them into Hessians
        output = []
        for param in params:
            hessian = self._params_to_hessian(param)
            output.append(hessian)
        return output

    def _params_to_hessian(self, param: np.ndarray) -> np.ndarray:
        """Convert the parameters for the linear model into a Hessian

        Args:
            param: Coefficients of the linear model
        Returns:
            The harmonic terms expressed as a Hessian matrix
        """
        # Get the parameters
        n_coords = len(self.reference) * 3
        triu_inds = np.triu_indices(n_coords)
        off_diag_triu_inds = np.triu_indices(n_coords, k=1)

        # Assemble the hessian
        hessian = np.zeros((n_coords, n_coords))
        hessian[triu_inds] = param[n_coords:]  # The first n_coords terms are the linear part
        hessian[off_diag_triu_inds] /= 2
        hessian.T[triu_inds] = hessian[triu_inds]
        # v = np.sqrt(self.reference.get_masses()).repeat(3).reshape(-1, 1)
        # hessian /= np.dot(v, v.T)
        return hessian
