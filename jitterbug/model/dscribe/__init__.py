"""Energy models using `DScribe <https://singroup.github.io/dscribe/latest/index.html>`_"""
import ase
from dscribe.descriptors.mbtr import MBTR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
import numpy as np

from .globald import DScribeGlobalEnergyModel


def make_global_mbtr_model(ref_atoms: ase.Atoms, n_points: int = 8, cutoff: float = 6.) -> DScribeGlobalEnergyModel:
    """Make an MBTR model using scikit-learn

    Args:
        ref_atoms: Reference atoms to use for the model
        n_points: Number of points to include in the MBTR grid
        cutoff: Cutoff distance for the descriptors (units: Angstrom)
    Returns:
        Energy model, ready to be trained
    """
    species = list(set(ref_atoms.get_chemical_symbols()))
    desc = MBTR(
        species=species,
        geometry={"function": "angle"},
        grid={"min": 0., "max": 180, "n": n_points, "sigma": 180. / n_points / 2.},
        weighting={"function": "smooth_cutoff", "r_cut": cutoff, "threshold": 1e-3},
        periodic=False,
    )
    model = Pipeline(
        [('scale', StandardScaler()),
         ('krr', GridSearchCV(KernelRidge(kernel='rbf', alpha=1e-10),
                              {'gamma': np.logspace(-5, 5, 32)}))]
    )
    return DScribeGlobalEnergyModel(
        reference=ref_atoms,
        model=model,
        descriptors=desc,
        num_calculators=2
    )
