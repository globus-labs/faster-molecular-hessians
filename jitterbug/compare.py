"""Tools for assessing the quality of a Hessian compared to a true one"""
from dataclasses import dataclass

import ase
import numpy as np
from ase.vibrations import VibrationsData


@dataclass
class HessianQuality:
    """Measurements of the quality of a Hessian"""

    # Thermodynamics
    zpe: float
    """Zero point energy (eV)"""
    zpe_error: float
    """Different between the ZPE and the target one"""

    # Vibrations
    vib_freqs: list[float]
    """Vibrational frequencies for our hessian (units: cm^-1)"""
    vib_errors: list[float]
    """Error between each frequency and the corresponding mode in the known hessian"""
    vib_mae: float
    """Mean absolute error for the vibrational modes"""


def compare_hessians(atoms: ase.Atoms, known_hessian: np.ndarray, approx_hessian: np.ndarray) -> HessianQuality:
    """Compare two different hessians for same atomic structure

    Args:
        atoms: Structure
        known_hessian: 2D form of the target Hessian
        approx_hessian: 2D form of an approximate Hessian
    Returns:
        Collection of the performance metrics
    """

    # Start by making a vibration data object
    known_vibs: VibrationsData = VibrationsData.from_2d(atoms, known_hessian)
    approx_vibs: VibrationsData = VibrationsData.from_2d(atoms, approx_hessian)

    # Compare the vibrational frequencies on the non-zero modes
    known_freqs = known_vibs.get_frequencies()
    is_real = np.isreal(known_freqs)
    approx_freqs = approx_vibs.get_frequencies()
    freq_error = np.subtract(approx_freqs[is_real], known_freqs[is_real])
    freq_mae = np.abs(freq_error).mean()

    # Assemble into a result object
    return HessianQuality(
        zpe=approx_vibs.get_zero_point_energy(),
        zpe_error=(approx_vibs.get_zero_point_energy() - known_vibs.get_zero_point_energy()),
        vib_freqs=np.real(approx_freqs[is_real]).tolist(),
        vib_errors=np.abs(freq_error),
        vib_mae=freq_mae
    )
