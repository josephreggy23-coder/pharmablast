import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    DEEP_PURPLE,
    FIGURE_DIR,
    GRAY,
    LIGHT_GRAY,
    LIGHT_PURPLE,
    POLYMER_QUANTUM_YIELDS,
    PURPLE,
    SOFT_PURPLE,
)
from .optimization import baseline_vs_improved
from .physics_model import drug_modulation, langmuir_saturation, predict_fluorescence_power_w, wavelength_absorption
from .sensor_model import sensor_response
from .utils import average_validation_error, save_validation_data, validation_dataframe


def _style_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=LIGHT_GRAY, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(LIGHT_GRAY)
    ax.tick_params(colors="#374151", labelsize=10)
    ax.title.set_color("#111827")
    ax.xaxis.label.set_color("#111827")
    ax.yaxis.label.set_color("#111827")


def _save(fig, name: str) -> list[str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / f"{name}.png"
    svg = FIGURE_DIR / f"{name}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [str(png), str(svg)]


def _predicted_mV(polymer="PE", diameter_um=500, dye=4, wavelength=550, count=80, alpha=0.0, pharma=0.0, excitation=1.0):
    power = predict_fluorescence_power_w(polymer, count, diameter_um, dye, wavelength, excitation, pharma, alpha)
    return float(sensor_response(power, include_random=False)["ideal_voltage_mV"])


def plot_saturation_curve() -> list[str]:
    x = np.linspace(0.5, 10.0, 400)
    y = langmuir_saturation(x)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvspan(3, 5, color=LIGHT_PURPLE, alpha=0.9, label="Useful operating range")
    ax.axvspan(5, 10, color=SOFT_PURPLE, alpha=0.75, label="Saturation zone")
    ax.plot(x, y, color=PURPLE, linewidth=3)
    ax.set_title("Langmuir Saturation Curve", fontsize=16, weight="bold")
    ax.set_xlabel("Nile Red concentration (ug/mL)", fontsize=12)
    ax.set_ylabel("Saturation factor S(cd)", fontsize=12)
    ax.set_xlim(0.5, 10)
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=True, facecolor="white", edgecolor=LIGHT_GRAY)
    _style_axes(ax)
    return _save(fig, "saturation_curve")


def plot_polymer_fluorescence_comparison() -> list[str]:
    polymers = ["PS", "PE", "PP", "PET"]
    values = [_predicted_mV(polymer=p, diameter_um=500, dye=4, wavelength=550, count=80) for p in polymers]
    labels = [f"{p}\nQp={POLYMER_QUANTUM_YIELDS[p]:.2f}" for p in polymers]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=[DEEP_PURPLE, PURPLE, "#a78bfa", "#c4b5fd"], edgecolor=DEEP_PURPLE, linewidth=0.8)
    ax.set_title("Polymer Fluorescence Comparison", fontsize=16, weight="bold")
    ax.set_ylabel("Predicted sensor signal (mV)", fontsize=12)
    ax.text(0.02, 0.95, "Higher quantum yield produces a brighter modeled signal under the same condition.", transform=ax.transAxes, va="top", color=GRAY)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025, f"{value:.0f}", ha="center", fontsize=10, color="#111827")
    _style_axes(ax)
    return _save(fig, "polymer_fluorescence_comparison")


def plot_wavelength_efficiency_curve() -> list[str]:
    x = np.linspace(400, 650, 500)
    y = wavelength_absorption(x)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color=PURPLE, linewidth=3)
    for wavelength in [450, 488, 550]:
        efficiency = float(wavelength_absorption(wavelength))
        ax.axvline(wavelength, color=DEEP_PURPLE if wavelength == 550 else "#a78bfa", linewidth=1.6, alpha=0.9)
        ax.scatter([wavelength], [efficiency], color=DEEP_PURPLE, s=55, zorder=4)
        ax.text(wavelength + 3, efficiency + 0.03, f"{wavelength} nm", color="#111827", fontsize=10)
    ax.set_title("Wavelength Efficiency Curve", fontsize=16, weight="bold")
    ax.set_xlabel("Excitation wavelength (nm)", fontsize=12)
    ax.set_ylabel("Absorption efficiency A(lambda)", fontsize=12)
    ax.set_xlim(400, 650)
    ax.set_ylim(0, 1.08)
    _style_axes(ax)
    return _save(fig, "wavelength_efficiency_curve")


def plot_particle_size_effect() -> list[str]:
    diameter = np.linspace(20, 1000, 400)
    signal = [_predicted_mV(diameter_um=d, dye=4, wavelength=550, count=80) for d in diameter]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(diameter, signal, color=PURPLE, linewidth=3)
    ax.fill_between(diameter, signal, color=LIGHT_PURPLE, alpha=0.65)
    ax.set_title("Particle Size Effect", fontsize=16, weight="bold")
    ax.set_xlabel("Particle diameter (um)", fontsize=12)
    ax.set_ylabel("Predicted fluorescence signal (mV)", fontsize=12)
    ax.text(0.03, 0.93, "The surface-area term scales with diameter squared for the spherical-particle approximation.", transform=ax.transAxes, color=GRAY, va="top")
    _style_axes(ax)
    return _save(fig, "particle_size_effect")


def plot_pharmaceutical_modulation() -> list[str]:
    concentration = np.linspace(0, 10, 300)
    quench = drug_modulation(concentration, -0.04)
    enhance = drug_modulation(concentration, 0.04)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(concentration, quench, color="#6d28d9", linewidth=3, label="Quenching example, alpha = -0.04")
    ax.plot(concentration, enhance, color="#a855f7", linewidth=3, label="Enhancement example, alpha = +0.04")
    ax.axhline(1.0, color=GRAY, linewidth=1.2, linestyle="--")
    ax.fill_between(concentration, quench, 1.0, color=LIGHT_PURPLE, alpha=0.6)
    ax.fill_between(concentration, 1.0, enhance, color=SOFT_PURPLE, alpha=0.85)
    ax.set_title("Pharmaceutical Modulation", fontsize=16, weight="bold")
    ax.set_xlabel("Pharmaceutical concentration (model units)", fontsize=12)
    ax.set_ylabel("Modulation factor Mdrug", fontsize=12)
    ax.text(0.03, 0.08, "Modeled effect: Mdrug = 1 + alpha * Cdrug", transform=ax.transAxes, color=GRAY)
    ax.legend(frameon=True, facecolor="white", edgecolor=LIGHT_GRAY)
    _style_axes(ax)
    return _save(fig, "pharmaceutical_modulation")


def plot_sensor_threshold() -> list[str]:
    counts = np.arange(1, 151)
    powers = np.array([predict_fluorescence_power_w("PE", c, 150, 2.0, 488, 1.0, 0.0, 0.0) for c in counts])
    sensor = sensor_response(powers, include_random=False)
    signal = sensor["ideal_voltage_mV"]
    threshold = sensor["detection_threshold_mV"]
    noise = sensor["total_noise_mV"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(counts, 0, threshold, color=SOFT_PURPLE, alpha=0.95, label="Low-signal region")
    ax.plot(counts, signal, color=PURPLE, linewidth=3, label="Modeled signal")
    ax.plot(counts, threshold, color="#ef4444", linewidth=2, linestyle="--", label="Detection threshold")
    ax.fill_between(counts, np.maximum(signal - noise, 0), signal + noise, color=LIGHT_PURPLE, alpha=0.8, label="Noise envelope")
    ax.set_title("Sensor Noise and Detection Threshold", fontsize=16, weight="bold")
    ax.set_xlabel("Particle count", fontsize=12)
    ax.set_ylabel("Sensor signal (mV)", fontsize=12)
    ax.legend(frameon=True, facecolor="white", edgecolor=LIGHT_GRAY)
    _style_axes(ax)
    return _save(fig, "sensor_threshold_plot")


def plot_baseline_vs_optimized() -> list[str]:
    df = baseline_vs_improved()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, metric, title, ylabel in [
        (axes[0], "fluorescence_mV", "Fluorescence Signal", "Predicted signal (mV)"),
        (axes[1], "signal_to_noise_ratio", "Signal-to-Noise Ratio", "SNR"),
    ]:
        ax.bar(df["condition"], df[metric], color=[LIGHT_PURPLE, PURPLE], edgecolor=DEEP_PURPLE, linewidth=0.8)
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=10)
        _style_axes(ax)
    fig.suptitle("Baseline vs Improved Detection Condition", fontsize=16, weight="bold")
    fig.text(0.5, 0.01, "This compares one reasonable improved condition, not a claimed global optimum.", ha="center", color=GRAY)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    return _save(fig, "baseline_vs_optimized")


def plot_validation() -> list[str]:
    df = save_validation_data()
    avg_error = average_validation_error()
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, df["experimental_mV"], width, yerr=df["experimental_sd_mV"], color=LIGHT_PURPLE, edgecolor=DEEP_PURPLE, label="Experimental")
    ax.bar(x + width / 2, df["simulated_mV"], width, yerr=df["simulated_sd_mV"], color=PURPLE, edgecolor=DEEP_PURPLE, label="Simulated")
    ax.set_xticks(x)
    ax.set_xticklabels(df["sample"], rotation=22, ha="right")
    ax.set_ylabel("Sensor signal (mV)", fontsize=12)
    ax.set_title("Validation Against Controlled PS/PE Fluorescence Values", fontsize=16, weight="bold")
    ax.text(0.02, 0.96, f"Average percent error = {avg_error:.2f}%", transform=ax.transAxes, va="top", color="#111827", fontsize=11, bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": LIGHT_GRAY})
    ax.legend(frameon=True, facecolor="white", edgecolor=LIGHT_GRAY)
    _style_axes(ax)
    return _save(fig, "validation_plot")


def generate_all_2d_figures() -> list[str]:
    """Generate all required 2D figures as PNG and SVG."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.titlesize": 15,
            "axes.labelsize": 12,
        }
    )
    paths = []
    for func in [
        plot_saturation_curve,
        plot_polymer_fluorescence_comparison,
        plot_wavelength_efficiency_curve,
        plot_particle_size_effect,
        plot_pharmaceutical_modulation,
        plot_sensor_threshold,
        plot_baseline_vs_optimized,
        plot_validation,
    ]:
        paths.extend(func())
    return paths
