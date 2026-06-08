import numpy as np
import pandas as pd
from scipy.stats import lognorm

from .config import (
    DYE_RANGE_UG_ML,
    EXCITATION_WAVELENGTHS_NM,
    GEOMETRIC_MEAN_DIAMETER_UM,
    GEOMETRIC_SIGMA,
    PARTICLE_DIAMETER_RANGE_UM,
    POLYMER_QUANTUM_YIELDS,
)
from .physics_model import model_factors, predict_fluorescence_power_w
from .sensor_model import sensor_response


def generate_synthetic_modeling_data(
    n_samples: int,
    seed: int,
    polymers: list[str] | None = None,
    particle_count_range: tuple[int, int] = (5, 150),
    dye_range: tuple[float, float] = DYE_RANGE_UG_ML,
    wavelengths: list[int] | None = None,
    pharmaceutical_range: tuple[float, float] = (0.0, 10.0),
) -> pd.DataFrame:
    """Generate physics-based synthetic modeling data without ML labels or predictions."""
    rng = np.random.default_rng(seed)
    polymers = polymers or list(POLYMER_QUANTUM_YIELDS)
    wavelengths = wavelengths or EXCITATION_WAVELENGTHS_NM

    polymer = rng.choice(polymers, size=n_samples)
    particle_count = rng.integers(particle_count_range[0], particle_count_range[1] + 1, size=n_samples)

    sigma = np.log(GEOMETRIC_SIGMA)
    diameter = lognorm(s=sigma, scale=GEOMETRIC_MEAN_DIAMETER_UM).rvs(size=n_samples, random_state=rng)
    diameter = np.clip(diameter, PARTICLE_DIAMETER_RANGE_UM[0], PARTICLE_DIAMETER_RANGE_UM[1])

    dye = rng.uniform(dye_range[0], dye_range[1], size=n_samples)
    wavelength = rng.choice(wavelengths, size=n_samples)
    excitation = rng.uniform(0.5, 1.5, size=n_samples)
    pharmaceutical = rng.uniform(pharmaceutical_range[0], pharmaceutical_range[1], size=n_samples)
    alpha = rng.uniform(-0.06, 0.06, size=n_samples)
    noise_level = rng.uniform(0.75, 1.35, size=n_samples)

    rows = []
    optical_power = np.empty(n_samples)
    for idx, poly in enumerate(polymer):
        optical_power[idx] = predict_fluorescence_power_w(
            poly,
            particle_count[idx],
            diameter[idx],
            dye[idx],
            wavelength[idx],
            excitation[idx],
            pharmaceutical[idx],
            alpha[idx],
        )

    sensor = sensor_response(optical_power, noise_level=noise_level, seed=seed, include_random=True)

    for idx, poly in enumerate(polymer):
        factors = model_factors(
            poly,
            particle_count[idx],
            diameter[idx],
            dye[idx],
            wavelength[idx],
            excitation[idx],
            pharmaceutical[idx],
            alpha[idx],
        )
        rows.append(
            {
                "polymer_type": poly,
                "quantum_yield": factors["quantum_yield"],
                "particle_count": int(particle_count[idx]),
                "particle_diameter_um": float(diameter[idx]),
                "particle_surface_area_m2": float(factors["surface_area_m2"]),
                "dye_concentration_ug_ml": float(dye[idx]),
                "saturation_factor": float(factors["saturation_factor"]),
                "excitation_wavelength_nm": int(wavelength[idx]),
                "wavelength_efficiency": float(factors["wavelength_efficiency"]),
                "excitation_intensity_relative": float(excitation[idx]),
                "pharmaceutical_concentration": float(pharmaceutical[idx]),
                "alpha_modulation_coefficient": float(alpha[idx]),
                "drug_modulation_factor": float(factors["drug_modulation_factor"]),
                "ideal_fluorescence_power_w": float(optical_power[idx]),
                "ideal_voltage_mV": float(sensor["ideal_voltage_mV"][idx]),
                "measured_voltage_mV": float(sensor["measured_voltage_mV"][idx]),
                "noise_std_mV": float(sensor["total_noise_mV"][idx]),
                "adc_counts": int(sensor["adc_counts"][idx]),
                "signal_to_noise_ratio": float(sensor["signal_to_noise_ratio"][idx]),
                "detection_threshold_mV": float(sensor["detection_threshold_mV"][idx]),
                "detected": bool(sensor["detected"][idx]),
            }
        )

    return pd.DataFrame(rows)
