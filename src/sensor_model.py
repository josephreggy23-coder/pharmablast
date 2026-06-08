import numpy as np

from .config import (
    ADC_BITS,
    ADC_REFERENCE_V,
    BACKGROUND_NOISE_MV,
    BOLTZMANN_J_K,
    DARK_CURRENT_A,
    DETECTION_SIGMA_MULTIPLIER,
    ELECTRON_CHARGE_C,
    FLICKER_NOISE_FRACTION,
    PHOTODIODE_RESPONSIVITY_A_PER_W,
    SENSOR_BANDWIDTH_HZ,
    TEMPERATURE_K,
    TRANSIMPEDANCE_GAIN_OHM,
)


def noise_components_mV(optical_power_w, noise_level=1.0) -> dict:
    """Estimate photodiode and electronics noise terms in millivolts."""
    optical_power = np.asarray(optical_power_w, dtype=float)
    signal_current_a = optical_power * PHOTODIODE_RESPONSIVITY_A_PER_W
    total_current_a = np.maximum(signal_current_a + DARK_CURRENT_A, 1e-15)

    shot_current_std = np.sqrt(2 * ELECTRON_CHARGE_C * total_current_a * SENSOR_BANDWIDTH_HZ)
    johnson_current_std = np.sqrt(4 * BOLTZMANN_J_K * TEMPERATURE_K * SENSOR_BANDWIDTH_HZ / TRANSIMPEDANCE_GAIN_OHM)
    flicker_current_std = FLICKER_NOISE_FRACTION * np.maximum(signal_current_a, DARK_CURRENT_A)

    shot_mV = shot_current_std * TRANSIMPEDANCE_GAIN_OHM * 1000.0
    johnson_mV = johnson_current_std * TRANSIMPEDANCE_GAIN_OHM * 1000.0
    flicker_mV = flicker_current_std * TRANSIMPEDANCE_GAIN_OHM * 1000.0
    background_mV = np.full_like(np.asarray(shot_mV), BACKGROUND_NOISE_MV, dtype=float)
    total_mV = np.sqrt(shot_mV**2 + johnson_mV**2 + flicker_mV**2 + background_mV**2) * noise_level
    return {
        "shot_noise_mV": shot_mV * noise_level,
        "johnson_noise_mV": johnson_mV * noise_level,
        "flicker_noise_mV": flicker_mV * noise_level,
        "background_noise_mV": background_mV * noise_level,
        "total_noise_mV": total_mV,
    }


def sensor_response(optical_power_w, noise_level=1.0, seed=None, include_random=True) -> dict:
    """Convert optical power to sensor voltage, ADC counts, SNR, and detection status."""
    rng = np.random.default_rng(seed)
    optical_power = np.asarray(optical_power_w, dtype=float)
    signal_current_a = optical_power * PHOTODIODE_RESPONSIVITY_A_PER_W
    ideal_voltage_mV = signal_current_a * TRANSIMPEDANCE_GAIN_OHM * 1000.0
    noise = noise_components_mV(optical_power, noise_level=noise_level)
    noise_std_mV = noise["total_noise_mV"]
    random_noise_mV = rng.normal(0.0, noise_std_mV) if include_random else np.zeros_like(ideal_voltage_mV)

    measured_mV = ideal_voltage_mV + random_noise_mV
    measured_v = np.clip(measured_mV / 1000.0, 0.0, ADC_REFERENCE_V)
    adc_levels = (2**ADC_BITS) - 1
    adc_counts = np.round(measured_v / ADC_REFERENCE_V * adc_levels)
    quantized_mV = adc_counts / adc_levels * ADC_REFERENCE_V * 1000.0

    threshold_mV = DETECTION_SIGMA_MULTIPLIER * noise_std_mV
    snr = ideal_voltage_mV / np.maximum(noise_std_mV, 1e-12)
    return {
        **noise,
        "ideal_voltage_mV": ideal_voltage_mV,
        "measured_voltage_mV": quantized_mV,
        "adc_counts": adc_counts,
        "signal_to_noise_ratio": snr,
        "detection_threshold_mV": threshold_mV,
        "detected": ideal_voltage_mV >= threshold_mV,
    }
