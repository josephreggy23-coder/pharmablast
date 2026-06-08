import numpy as np
import pandas as pd

from .config import (
    ABSORPTION_CENTER_NM,
    ABSORPTION_SD_NM,
    ADC_BITS,
    ADC_REFERENCE_V,
    BOLTZMANN_J_K,
    DRUG_ALPHAS,
    ELECTRON_CHARGE_C,
    EXCITATION_WAVELENGTHS_NM,
    GEOMETRIC_MEAN_DIAMETER_UM,
    GEOMETRIC_SIGMA,
    LANGMUIR_HALF_SAT_UG_ML,
    PHOTODIODE_RESPONSIVITY_A_PER_W,
    POLYMER_QUANTUM_YIELDS,
    SENSOR_BANDWIDTH_HZ,
    TEMPERATURE_K,
    TRANSIMPEDANCE_GAIN_OHM,
)


def gaussian_absorption(wavelength_nm: np.ndarray) -> np.ndarray:
    """Nile Red excitation weighting centered around 550 nm."""
    z = (wavelength_nm - ABSORPTION_CENTER_NM) / ABSORPTION_SD_NM
    return np.exp(-0.5 * z**2)


def langmuir_saturation(dye_concentration_ug_ml: np.ndarray) -> np.ndarray:
    """Langmuir dye binding saturation with half saturation near 2 ug/mL."""
    cd = np.asarray(dye_concentration_ug_ml)
    return cd / (LANGMUIR_HALF_SAT_UG_ML + cd)


def surface_area_scale(count: np.ndarray, mean_diameter_um: np.ndarray) -> np.ndarray:
    """Approximate exposed area as count times spherical particle area."""
    diameter_m = mean_diameter_um * 1e-6
    area_m2 = np.pi * diameter_m**2
    return count * area_m2


def electronics_response(optical_power_w: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert optical power to ADC counts with photodiode, TIA, and noise."""
    photocurrent_a = optical_power_w * PHOTODIODE_RESPONSIVITY_A_PER_W
    signal_v = photocurrent_a * TRANSIMPEDANCE_GAIN_OHM

    shot_current_std = np.sqrt(2 * ELECTRON_CHARGE_C * np.maximum(photocurrent_a, 1e-15) * SENSOR_BANDWIDTH_HZ)
    johnson_current_std = np.sqrt(4 * BOLTZMANN_J_K * TEMPERATURE_K * SENSOR_BANDWIDTH_HZ / TRANSIMPEDANCE_GAIN_OHM)
    flicker_current_std = 0.035 * np.maximum(photocurrent_a, 1e-15)

    total_current_std = np.sqrt(shot_current_std**2 + johnson_current_std**2 + flicker_current_std**2)
    noise_v = rng.normal(0.0, total_current_std * TRANSIMPEDANCE_GAIN_OHM)
    noisy_v = np.clip(signal_v + noise_v, 0.0, ADC_REFERENCE_V)

    adc_levels = (2**ADC_BITS) - 1
    adc_counts = np.round(noisy_v / ADC_REFERENCE_V * adc_levels)
    quantized_v = adc_counts / adc_levels * ADC_REFERENCE_V
    snr = signal_v / np.maximum(total_current_std * TRANSIMPEDANCE_GAIN_OHM, 1e-12)
    return quantized_v, adc_counts, snr


def generate_synthetic_dataset(n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    polymers = np.array(list(POLYMER_QUANTUM_YIELDS))
    polymer = rng.choice(polymers, size=n_samples, p=[0.28, 0.27, 0.24, 0.21])
    quantum_yield = np.array([POLYMER_QUANTUM_YIELDS[p] for p in polymer])

    pre_post = rng.choice(["pre_filtration", "post_filtration"], size=n_samples, p=[0.55, 0.45])
    particle_count_pre = rng.poisson(85, size=n_samples) + 1
    removal_fraction = rng.beta(7.0, 2.3, size=n_samples)
    particle_count = np.where(
        pre_post == "post_filtration",
        np.maximum(1, np.round(particle_count_pre * (1.0 - 0.72 * removal_fraction))).astype(int),
        particle_count_pre,
    )

    mean_diameter = rng.lognormal(np.log(GEOMETRIC_MEAN_DIAMETER_UM), np.log(GEOMETRIC_SIGMA), size=n_samples)
    mean_diameter = np.clip(mean_diameter, 20.0, 3_000.0)
    size_cv = rng.uniform(0.18, 0.72, size=n_samples)
    diameter_std = mean_diameter * size_cv
    diameter_median = mean_diameter / np.sqrt(1.0 + size_cv**2)

    dye_concentration = rng.uniform(0.05, 8.0, size=n_samples)
    wavelength = rng.choice(EXCITATION_WAVELENGTHS_NM, size=n_samples, p=[0.22, 0.33, 0.45])

    drug_type = rng.choice(list(DRUG_ALPHAS), size=n_samples, p=[0.50, 0.50])
    alpha = np.array([DRUG_ALPHAS[d] for d in drug_type])
    drug_concentration = rng.gamma(shape=2.0, scale=1.6, size=n_samples)
    drug_concentration = np.clip(drug_concentration, 0.0, 12.0)
    drug_modulation = np.clip(1.0 + alpha * drug_concentration, 0.55, 1.35)

    absorption = gaussian_absorption(wavelength)
    saturation = langmuir_saturation(dye_concentration)
    area = surface_area_scale(particle_count, mean_diameter)

    # Scale converts relative fluorescence into optical power in a sensor-like range.
    electronics_scale = 8.0e-2
    optical_power = quantum_yield * absorption * saturation * area * electronics_scale * drug_modulation
    optical_power *= rng.normal(1.0, 0.045, size=n_samples)
    optical_power = np.clip(optical_power, 1e-12, None)

    measured_v, adc_counts, snr = electronics_response(optical_power, rng)
    fluorescence_intensity = measured_v / TRANSIMPEDANCE_GAIN_OHM / PHOTODIODE_RESPONSIVITY_A_PER_W
    # A calibrated pure-polymer reference measurement estimates Qp with noise.
    estimated_quantum_yield = np.clip(quantum_yield + rng.normal(0.0, 0.0095, size=n_samples), 0.25, 0.50)

    return pd.DataFrame(
        {
            "polymer_type": polymer,
            "quantum_yield": quantum_yield,
            "estimated_quantum_yield": estimated_quantum_yield,
            "particle_count": particle_count,
            "dye_concentration_ug_ml": dye_concentration,
            "excitation_wavelength_nm": wavelength,
            "particle_mean_diameter_um": mean_diameter,
            "particle_median_diameter_um": diameter_median,
            "particle_std_diameter_um": diameter_std,
            "surface_area_scale_m2": area,
            "absorption_factor": absorption,
            "saturation_fraction": saturation,
            "drug_type": drug_type,
            "drug_concentration_ug_l": drug_concentration,
            "drug_alpha": alpha,
            "drug_modulation": drug_modulation,
            "filtration_state": pre_post,
            "optical_power_w": optical_power,
            "fluorescence_intensity_w": fluorescence_intensity,
            "sensor_voltage_v": measured_v,
            "adc_counts": adc_counts,
            "signal_to_noise_ratio": snr,
        }
    )
