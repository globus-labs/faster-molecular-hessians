"""Models which treat each term in the Hessian as independent

We achieve a approximate Hessian by conditioning a linear model
on the assumption that the Hessian contains parameters which are zero
and that smaller parameters are more likely than larger
"""
import numpy as np
from ase import Atoms
from ase import io as aseio
from geometric.molecule import Molecule
from geometric.internal import DelocalizedInternalCoordinates as DIC
from sklearn.linear_model import ARDRegression
from sklearn.linear_model._base import LinearModel
from .base import EnergyModel
import geometric


def get_internal_diff(ref_int: DIC, ref_mol: Molecule, atoms: Atoms) -> np.ndarray:
    """Finds the differences in the values of the internal coordinates for two
    of geometries.
    Args:
        ref_int: delocalized internal coordinate object (geometric package)
        ref_mol: molecule object of minimum (geometric package)
        atoms: molecule object of perturbation structure (ASE package)
    Returns:
        Array of displacements, in internal coordinates
    """
    # convert ASE to geometric molecule object
    # this i/o is major slowdown, there has to be a better way
    filename = 'tmp.xyz'
    aseio.write(filename, atoms, 'xyz')
    pert_mol = Molecule(filename, 'xyz')

    ref_coords = ref_mol.xyzs[0].flatten()
    pert_coords = pert_mol.xyzs[0].flatten()
    int_diff = ref_int.calcDiff(pert_coords, ref_coords)
    return (int_diff)

# def get_internal_coords(atoms: Atoms) -> np.ndarray:
#     #filename = 'tmp.xyz'
#     #aseio.write(filename, atoms, 'xyz')
#     #molecule = geometric.molecule.Molecule(filename, 'xyz')
#     #IC = geometric.internal.DelocalizedInternalCoordinates(molecule, build=True, remove_tr=True)
#     #coords = molecule.xyzs[0].flatten()#*geometric.nifty.ang2bohr
#     #print('without rts')
#     #print(IC.Prims)
#     #print(IC.calculate(coords))
#     ##IC.remove_TR(coords)
#     ##print('without rts')
#     ##print(IC.Prims)
#     ##print(IC.calculate(coords))
#     #return IC.calculate(coords)


def get_model_internal_inputs(atoms: Atoms, ref_mol: Molecule, ref_IC: DIC) -> np.ndarray:
    """Sets up first and second-order displacement vectors for a perturbation geometry
    Args:
        atoms: molecule object of perturbation structure (ASE package)
        ref_mol: molecule object of minimum (geometric package)
        ref_IC: delocalized internal coordinate object (geometric package)
    Returns:
        Array of Jacobian and Hessian displacements, in internal coordinates
    """
    # Compute the displacements and the products of displacement
    disp_matrix = get_internal_diff(ref_IC, ref_mol, atoms)
    disp_prod_matrix = disp_matrix[:, None] * disp_matrix[None, :]
    n_terms = len(atoms) * 3 - 6
    off_diag = np.triu_indices(n_terms, k=1)
    disp_prod_matrix[off_diag] *= 2.
    # Append the displacements and products of displacements
    return np.concatenate([
        disp_matrix,
        disp_prod_matrix[np.triu_indices(n_terms)] / 2
    ], axis=0)


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
    disp_prod_matrix[off_diag] *= 2.
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
        # Convert ASE object to geometric molecule object
        filename = 'tmp.xyz'
        aseio.write(filename, self.reference, 'xyz')
        molecule = Molecule(filename, 'xyz')
        # Build internal coordinates object
        IC = DIC(molecule, build=True, remove_tr=True)
        # X: Displacement vectors for each
        x = [get_model_internal_inputs(atoms, molecule, IC) for atoms in data]
        # Y: Subtract off the reference energy
        ref_energy = self.reference.get_potential_energy()
        y = [atoms.get_potential_energy() - ref_energy for atoms in data]
        # Fit the ARD model and ensure it captures the data well
        model = self.regressor(fit_intercept=True).fit(x, y)
        pred = model.predict(x)
        max_error = np.abs(pred - y).max()
        if max_error > 0.002:
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
        filename = 'tmp.xyz'
        aseio.write(filename, self.reference, 'xyz')
        n_coords = len(self.reference) * 3 - 6
        triu_inds = np.triu_indices(n_coords)
        off_diag_triu_inds = np.triu_indices(n_coords, k=1)

        # Assemble the hessian
        hessian = np.zeros((n_coords, n_coords))
        gradq = np.zeros(n_coords)
        gradq = param[:n_coords]
        hessian[triu_inds] = param[n_coords:]  # The first n_coords terms are the linear part
        hessian[off_diag_triu_inds] /= 2
        hessian.T[triu_inds] = hessian[triu_inds]
        molecule = Molecule(filename, 'xyz')
        IC = DIC(molecule, build=True, remove_tr=True)
        coords = molecule.xyzs[0].flatten()*geometric.nifty.ang2bohr
        hessian_cart = IC.calcHessCart(coords,  gradq, hessian)
        # print(IC.Internals)
        # print(IC.Prims)
        # print(geometric.normal_modes.frequency_analysis(coords, hessian_cart, elem = molecule.elem))
        return hessian_cart
