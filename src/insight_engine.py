from dataclasses import dataclass

import numpy as np

from .config import EXCITATION_WAVELENGTHS_NM, POLYMER_QUANTUM_YIELDS
from .physics_model import model_factors, predict_fluorescence_power_w
from .sensor_model import sensor_response


@dataclass(frozen=True)
class Condition:
    polymer: str
    particle_count: int
    diameter_um: float
    dye_concentration_ug_ml: float
    wavelength_nm: int
    excitation_intensity: float
    pharmaceutical_concentration: float
    alpha: float
    noise_level: float


def _evaluate(condition: Condition) -> dict:
    power = predict_fluorescence_power_w(
        condition.polymer,
        condition.particle_count,
        condition.diameter_um,
        condition.dye_concentration_ug_ml,
        condition.wavelength_nm,
        condition.excitation_intensity,
        condition.pharmaceutical_concentration,
        condition.alpha,
    )
    sensor = sensor_response(power, noise_level=condition.noise_level, include_random=False)
    factors = model_factors(
        condition.polymer,
        condition.particle_count,
        condition.diameter_um,
        condition.dye_concentration_ug_ml,
        condition.wavelength_nm,
        condition.excitation_intensity,
        condition.pharmaceutical_concentration,
        condition.alpha,
    )
    return {"sensor": sensor, "factors": factors}


def detection_score(snr: float, saturation: float, wavelength_efficiency: float, drug_modulation: float) -> float:
    """Return a 0-100 score from interpretable model quality signals."""
    snr_score = min(1.0, max(0.0, snr / 25.0))
    saturation_score = 1.0 - min(1.0, abs(saturation - 0.67) / 0.67)
    wavelength_score = min(1.0, max(0.0, wavelength_efficiency))
    drug_score = min(1.0, max(0.0, drug_modulation / 1.25))
    score = 100.0 * (0.48 * snr_score + 0.22 * saturation_score + 0.20 * wavelength_score + 0.10 * drug_score)
    return float(np.clip(score, 0.0, 100.0))


def score_label(score: float) -> str:
    if score >= 82:
        return "Strong"
    if score >= 62:
        return "Usable"
    if score >= 40:
        return "Marginal"
    return "Weak"


def diagnose_condition(condition: Condition) -> dict:
    """Generate rule-based expert guidance from the physics model."""
    evaluated = _evaluate(condition)
    sensor = evaluated["sensor"]
    factors = evaluated["factors"]
    snr = float(sensor["signal_to_noise_ratio"])
    saturation = float(factors["saturation_factor"])
    wavelength_efficiency = float(factors["wavelength_efficiency"])
    drug_mod = float(factors["drug_modulation_factor"])
    score = detection_score(snr, saturation, wavelength_efficiency, drug_mod)

    limiting_factors = []
    recommendations = []

    if snr < 5:
        limiting_factors.append("Signal is close to the detection threshold")
        recommendations.append("Increase particle count, excitation intensity, or particle collection before interpreting this signal.")
    elif snr < 12:
        limiting_factors.append("SNR is usable but not comfortable")
        recommendations.append("Improve signal margin by moving wavelength and dye concentration closer to the high-response range.")

    if condition.dye_concentration_ug_ml < 3.0:
        limiting_factors.append("Dye concentration is below the useful operating range")
        recommendations.append("Move Nile Red concentration toward 3-5 ug/mL to increase saturation without relying on excess dye.")
    elif condition.dye_concentration_ug_ml > 5.0:
        limiting_factors.append("Dye concentration is in the saturation zone")
        recommendations.append("Reduce dye toward 3-5 ug/mL if the goal is efficient signal gain rather than maximum dye loading.")

    if condition.wavelength_nm != 550:
        limiting_factors.append("Excitation wavelength is not at the modeled Nile Red absorption peak")
        recommendations.append("Compare against 550 nm excitation because the Gaussian absorption model peaks there.")

    if condition.diameter_um < 250:
        limiting_factors.append("Small particle diameter limits surface-area-driven signal")
        recommendations.append("Use enrichment/collection steps or compare larger particle fractions when available.")

    if condition.alpha < -0.015 and condition.pharmaceutical_concentration > 1:
        limiting_factors.append("Pharmaceutical modulation is quenching the modeled signal")
        recommendations.append("Report this condition as a possible quenching case and compare it against a no-drug baseline.")
    elif condition.alpha > 0.015 and condition.pharmaceutical_concentration > 1:
        recommendations.append("Enhancement is increasing signal; compare against baseline so the effect is not mistaken for more particles.")

    if condition.noise_level > 1.25:
        limiting_factors.append("Sensor noise level is elevated")
        recommendations.append("Improve electronics shielding, averaging, or optical filtering in the modeled setup.")

    if not limiting_factors:
        limiting_factors.append("No major single limiting factor")
    if not recommendations:
        recommendations.append("This is a strong modeled condition; use it as a reference comparison rather than claiming global optimization.")

    if score >= 82:
        summary = "The model sees this as a strong detection condition with comfortable signal margin."
    elif score >= 62:
        summary = "The model sees this as a usable condition, but at least one parameter can still be improved."
    elif score >= 40:
        summary = "The model sees this as a marginal condition where interpretation needs caution."
    else:
        summary = "The model sees this as a weak detection condition dominated by low signal or poor operating settings."

    return {
        "score": score,
        "label": score_label(score),
        "summary": summary,
        "limiting_factors": limiting_factors,
        "recommendations": recommendations,
    }


def suggest_operating_condition(condition: Condition) -> dict:
    """Search a small interpretable grid for a better operating condition.

    This is a guided model recommendation, not a claim of global optimization.
    """
    dye_options = np.linspace(3.0, 5.0, 9)
    excitation_options = [condition.excitation_intensity, min(2.0, max(condition.excitation_intensity, 1.25))]
    candidates = []
    for dye in dye_options:
        for wavelength in EXCITATION_WAVELENGTHS_NM:
            for excitation in excitation_options:
                candidate = Condition(
                    polymer=condition.polymer,
                    particle_count=condition.particle_count,
                    diameter_um=condition.diameter_um,
                    dye_concentration_ug_ml=float(dye),
                    wavelength_nm=int(wavelength),
                    excitation_intensity=float(excitation),
                    pharmaceutical_concentration=condition.pharmaceutical_concentration,
                    alpha=condition.alpha,
                    noise_level=condition.noise_level,
                )
                evaluated = _evaluate(candidate)
                sensor = evaluated["sensor"]
                factors = evaluated["factors"]
                score = detection_score(
                    float(sensor["signal_to_noise_ratio"]),
                    float(factors["saturation_factor"]),
                    float(factors["wavelength_efficiency"]),
                    float(factors["drug_modulation_factor"]),
                )
                candidates.append((score, candidate, sensor, factors))

    candidates.sort(key=lambda row: row[0], reverse=True)
    best_score, best_condition, best_sensor, best_factors = candidates[0]
    return {
        "condition": best_condition,
        "score": best_score,
        "label": score_label(best_score),
        "signal_mV": float(best_sensor["ideal_voltage_mV"]),
        "snr": float(best_sensor["signal_to_noise_ratio"]),
        "saturation_factor": float(best_factors["saturation_factor"]),
        "wavelength_efficiency": float(best_factors["wavelength_efficiency"]),
    }


def compare_polymers_for_condition(condition: Condition) -> list[dict]:
    """Compare polymer brightness under the same user-selected condition."""
    rows = []
    for polymer in POLYMER_QUANTUM_YIELDS:
        candidate = Condition(
            polymer=polymer,
            particle_count=condition.particle_count,
            diameter_um=condition.diameter_um,
            dye_concentration_ug_ml=condition.dye_concentration_ug_ml,
            wavelength_nm=condition.wavelength_nm,
            excitation_intensity=condition.excitation_intensity,
            pharmaceutical_concentration=condition.pharmaceutical_concentration,
            alpha=condition.alpha,
            noise_level=condition.noise_level,
        )
        evaluated = _evaluate(candidate)
        rows.append(
            {
                "polymer": polymer,
                "quantum_yield": POLYMER_QUANTUM_YIELDS[polymer],
                "signal_mV": float(evaluated["sensor"]["ideal_voltage_mV"]),
                "snr": float(evaluated["sensor"]["signal_to_noise_ratio"]),
            }
        )
    return sorted(rows, key=lambda row: row["signal_mV"], reverse=True)
