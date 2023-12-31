"""Tools for assessing the quality of a Hessian compared to a true one"""
from dataclasses import dataclass
from typing import Optional

import ase
from ase import units
import numpy as np
from ase.vibrations import VibrationsData
from pmutt.statmech import StatMech, presets


@dataclass
class HessianQuality:
    """Measurements of the quality of a Hessian"""

    # Metadata
    scale_factor: float
    """Scaling factor used for frequencies"""

    # Thermodynamics
    zpe: float
    """Zero point energy (kcal/mol)"""
    zpe_error: float
    """Different between the ZPE and the target one"""
    cp: list[float]
    """Heat capacity as a function of temperature (units: kcal/mol/K)"""
    cp_error: list[float]
    """Difference between known and approximate heat capacity as a function of temperature (units: kcal/mol/K)"""
    h: list[float]
    """Enthalpy as a function of temperature (units: kcal/mol)"""
    h_error: list[float]
    """Error between known and approximate enthalpy as a function of temperature (units: kcal/mol)"""
    temps: list[float]
    """Temperatures at which Cp was evaluated (units: K)"""

    # Vibrations
    vib_freqs: list[float]
    """Vibrational frequencies for our hessian (units: cm^-1)"""
    vib_errors: list[float]
    """Error between each frequency and the corresponding mode in the known hessian"""
    vib_mae: float
    """Mean absolute error for the vibrational modes"""


def compare_hessians(atoms: ase.Atoms, known_hessian: np.ndarray, approx_hessian: np.ndarray, scale_factor: Optional[float] = 1.) -> HessianQuality:
    """Compare two different hessians for same atomic structure

    Args:
        atoms: Structure
        known_hessian: 2D form of the target Hessian
        approx_hessian: 2D form of an approximate Hessian
        scale_factor: Factor by which to scale frequencies from approximate Hessian before comparison.
            Set to ``None`` to use the median ratio between the approximate and known frequency from each mode.
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

    # Scale, if desired
    if scale_factor is None:
        scale_factor = np.median(np.divide(known_freqs, approx_freqs))
    approx_freqs *= scale_factor

    freq_error = np.subtract(approx_freqs[is_real], known_freqs[is_real])
    freq_mae = np.abs(freq_error).mean()

    # Compare the enthalpy and heat capacity
    #  TODO (wardlt): Might actually want to compute the symmetry number
    known_harm = StatMech(vib_wavenumbers=np.real(known_freqs[is_real]), atoms=atoms, symmetrynumber=1, **presets['harmonic'])
    approx_harm = StatMech(vib_wavenumbers=np.real(approx_freqs[is_real]), atoms=atoms, symmetrynumber=1, **presets['harmonic'])

    approx_zpe = approx_harm.vib_model.get_ZPE() * units.mol / units.kcal
    known_zpe = known_harm.vib_model.get_ZPE() * units.mol / units.kcal
    zpe_error = approx_zpe - known_zpe

    temps = np.linspace(1., 373, 128)
    known_cp = np.array([known_harm.get_Cp('kcal/mol/K', T=t) for t in temps])
    approx_cp = np.array([approx_harm.get_Cp('kcal/mol/K', T=t) for t in temps])
    known_h = np.array([known_harm.get_H('kcal/mol', T=t) for t in temps])
    approx_h = np.array([approx_harm.get_H('kcal/mol', T=t) for t in temps])

    # Assemble into a result object
    return HessianQuality(
        scale_factor=scale_factor,
        zpe=approx_zpe,
        zpe_error=zpe_error,
        vib_freqs=np.real(approx_freqs[is_real]).tolist(),
        vib_errors=np.abs(freq_error),
        vib_mae=freq_mae,
        cp=approx_cp.tolist(),
        cp_error=(known_cp - approx_cp).tolist(),
        h=approx_h,
        h_error=(known_h - approx_h).tolist(),
        temps=temps.tolist()
    )
