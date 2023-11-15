"""Tools for assessing the quality of a Hessian compared to a true one"""
from dataclasses import dataclass

import ase
from ase import units
import numpy as np
from ase.vibrations import VibrationsData
from pmutt.statmech import StatMech, presets


@dataclass
class HessianQuality:
    """Measurements of the quality of a Hessian"""

    # Thermodynamics
    zpe: float
    """Zero point energy (kcal/mol)"""
    zpe_error: float
    """Different between the ZPE and the target one"""
    cp: list[float]
    """Heat capacity as a function of temperature (units: kcal/mol/K)"""
    cp_error: list[float]
    """Difference between known and approximate heat capacity as a function of temperature (units: kcal/mol/K)"""
    temps: list[float]
    """Temperatures at which Cp was evaluated (units: K)"""

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

    # Compare the enthalpy and heat capacity
    known_harm = StatMech(vib_wavenumbers=np.real(known_freqs[is_real]), atoms=atoms, symmetrynumber=1, **presets['harmonic'])
    approx_harm = StatMech(vib_wavenumbers=np.real(approx_freqs[is_real]), atoms=atoms, symmetrynumber=1, **presets['harmonic'])

    temps = np.linspace(1., 373, 128)
    known_cp = np.array([known_harm.get_Cp('kcal/mol/K', T=t) for t in temps])
    approx_cp = np.array([approx_harm.get_Cp('kcal/mol/K', T=t) for t in temps])

    # Assemble into a result object
    return HessianQuality(
        zpe=approx_vibs.get_zero_point_energy() * units.mol / units.kcal,
        zpe_error=(approx_vibs.get_zero_point_energy() - known_vibs.get_zero_point_energy()) * units.mol / units.kcal,
        vib_freqs=np.real(approx_freqs[is_real]).tolist(),
        vib_errors=np.abs(freq_error),
        vib_mae=freq_mae,
        cp=approx_cp.tolist(),
        cp_error=(known_cp - approx_cp).tolist(),
        temps=temps.tolist()
    )
