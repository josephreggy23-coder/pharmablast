import numpy as np

from .config import (
    ABSORPTION_CENTER_NM,
    ABSORPTION_SD_NM,
    EXCITATION_TO_POWER_SCALE,
    LANGMUIR_HALF_SAT_UG_ML,
    POLYMER_QUANTUM_YIELDS,
    REFERENCE_EXCITATION_INTENSITY,
)


def quantum_yield(polymer: str) -> float:
    """Return the polymer-specific fluorescence quantum yield."""
    try:
        return POLYMER_QUANTUM_YIELDS[polymer]
    except KeyError as exc:
        allowed = ", ".join(POLYMER_QUANTUM_YIELDS)
        raise ValueError(f"Unknown polymer {polymer!r}. Choose one of: {allowed}.") from exc


def wavelength_absorption(wavelength_nm):
    """Gaussian Nile Red absorption efficiency centered at 550 nm."""
    wavelength = np.asarray(wavelength_nm, dtype=float)
    z = (wavelength - ABSORPTION_CENTER_NM) / ABSORPTION_SD_NM
    return np.exp(-0.5 * z**2)


def langmuir_saturation(dye_concentration_ug_ml):
    """Langmuir dye saturation using a half-saturation constant near 2 ug/mL."""
    concentration = np.asarray(dye_concentration_ug_ml, dtype=float)
    concentration = np.clip(concentration, 0.0, None)
    return concentration / (LANGMUIR_HALF_SAT_UG_ML + concentration)


def particle_surface_area_m2(diameter_um, particle_count=1):
    """Approximate total surface area using spherical particles."""
    diameter_m = np.asarray(diameter_um, dtype=float) * 1e-6
    count = np.asarray(particle_count, dtype=float)
    return count * np.pi * diameter_m**2


def drug_modulation(pharmaceutical_concentration, alpha):
    """Pharmaceutical modulation term Mdrug = 1 + alpha * Cdrug."""
    modulation = 1.0 + np.asarray(alpha, dtype=float) * np.asarray(pharmaceutical_concentration, dtype=float)
    return np.clip(modulation, 0.05, None)


def predict_fluorescence_power_w(
    polymer: str,
    particle_count,
    diameter_um,
    dye_concentration_ug_ml,
    wavelength_nm,
    excitation_intensity=1.0,
    pharmaceutical_concentration=0.0,
    alpha=0.0,
):
    """Predict fluorescence optical power from the project equation.

    I = Qp * A(lambda) * S(cd) * As * E * Mdrug
    """
    q_p = quantum_yield(polymer)
    absorption = wavelength_absorption(wavelength_nm)
    saturation = langmuir_saturation(dye_concentration_ug_ml)
    area = particle_surface_area_m2(diameter_um, particle_count)
    excitation = np.asarray(excitation_intensity, dtype=float) / REFERENCE_EXCITATION_INTENSITY
    modulation = drug_modulation(pharmaceutical_concentration, alpha)
    return q_p * absorption * saturation * area * excitation * modulation * EXCITATION_TO_POWER_SCALE


def model_factors(
    polymer: str,
    particle_count,
    diameter_um,
    dye_concentration_ug_ml,
    wavelength_nm,
    excitation_intensity=1.0,
    pharmaceutical_concentration=0.0,
    alpha=0.0,
) -> dict:
    """Return each physics factor so users can inspect the model."""
    q_p = quantum_yield(polymer)
    absorption = wavelength_absorption(wavelength_nm)
    saturation = langmuir_saturation(dye_concentration_ug_ml)
    area = particle_surface_area_m2(diameter_um, particle_count)
    excitation = np.asarray(excitation_intensity, dtype=float) / REFERENCE_EXCITATION_INTENSITY
    modulation = drug_modulation(pharmaceutical_concentration, alpha)
    intensity = q_p * absorption * saturation * area * excitation * modulation * EXCITATION_TO_POWER_SCALE
    return {
        "quantum_yield": q_p,
        "wavelength_efficiency": absorption,
        "saturation_factor": saturation,
        "surface_area_m2": area,
        "excitation_factor": excitation,
        "drug_modulation_factor": modulation,
        "fluorescence_power_w": intensity,
    }
