import pandas as pd

from .physics_model import model_factors, predict_fluorescence_power_w
from .sensor_model import sensor_response


BASELINE_CONDITION = {
    "polymer": "PE",
    "particle_count": 50,
    "diameter_um": 250.0,
    "dye_concentration_ug_ml": 1.0,
    "wavelength_nm": 488,
    "excitation_intensity": 1.0,
    "pharmaceutical_concentration": 2.0,
    "alpha": -0.02,
}

IMPROVED_CONDITION = {
    "polymer": "PE",
    "particle_count": 50,
    "diameter_um": 500.0,
    "dye_concentration_ug_ml": 4.0,
    "wavelength_nm": 550,
    "excitation_intensity": 1.0,
    "pharmaceutical_concentration": 2.0,
    "alpha": -0.02,
}


def evaluate_condition(label: str, condition: dict) -> dict:
    """Evaluate one user-readable detection condition."""
    power = predict_fluorescence_power_w(**condition)
    sensor = sensor_response(power, include_random=False)
    factors = model_factors(**condition)
    return {
        "condition": label,
        "fluorescence_mV": float(sensor["ideal_voltage_mV"]),
        "signal_to_noise_ratio": float(sensor["signal_to_noise_ratio"]),
        "saturation_factor": float(factors["saturation_factor"]),
        "wavelength_efficiency": float(factors["wavelength_efficiency"]),
        "drug_modulation_factor": float(factors["drug_modulation_factor"]),
        "detected": bool(sensor["detected"]),
    }


def baseline_vs_improved() -> pd.DataFrame:
    """Compare a baseline condition with a reasonable improved condition."""
    return pd.DataFrame(
        [
            evaluate_condition("Baseline", BASELINE_CONDITION),
            evaluate_condition("Improved operating condition", IMPROVED_CONDITION),
        ]
    )
