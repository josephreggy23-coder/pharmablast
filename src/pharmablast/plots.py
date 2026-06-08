import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.metrics import ConfusionMatrixDisplay

from .config import FIGURE_DIR, POLYMER_QUANTUM_YIELDS
from .simulation import gaussian_absorption, langmuir_saturation


def save_saturation_curve() -> None:
    cd = np.linspace(0.01, 8.0, 300)
    plt.figure(figsize=(7, 5))
    plt.plot(cd, langmuir_saturation(cd), color="#0b6e69", linewidth=2.5)
    plt.xlabel("Nile Red dye concentration (ug/mL)")
    plt.ylabel("Langmuir saturation fraction")
    plt.title("Dye Saturation Curve")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "saturation_curve.png", dpi=180)
    plt.close()


def save_polymer_comparison() -> None:
    wavelengths = np.array([450, 488, 550])
    plt.figure(figsize=(7, 5))
    for polymer, qy in POLYMER_QUANTUM_YIELDS.items():
        plt.plot(wavelengths, qy * gaussian_absorption(wavelengths), marker="o", linewidth=2, label=polymer)
    plt.xlabel("Excitation wavelength (nm)")
    plt.ylabel("Relative fluorescence response")
    plt.title("Polymer Fluorescence Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "polymer_fluorescence_comparison.png", dpi=180)
    plt.close()


def save_confusion_matrix(result: dict) -> None:
    display = ConfusionMatrixDisplay(confusion_matrix=result["confusion_matrix"], display_labels=result["labels"])
    display.plot(cmap="Blues", values_format="d")
    plt.title("Polymer Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=180)
    plt.close()


def save_predicted_vs_actual(regression_result: dict) -> None:
    y = regression_result["y_test"]
    pred = regression_result["pred"]
    plt.figure(figsize=(6, 6))
    plt.scatter(y, pred, s=12, alpha=0.35, color="#4357ad")
    low, high = min(y.min(), pred.min()), max(y.max(), pred.max())
    plt.plot([low, high], [low, high], color="#c44536", linewidth=2)
    plt.xlabel("Actual drug concentration (ug/L)")
    plt.ylabel("Predicted drug concentration (ug/L)")
    plt.title("Predicted vs Actual Concentration")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "predicted_vs_actual_concentration.png", dpi=180)
    plt.close()


def save_feature_importance(importance: pd.DataFrame) -> None:
    top = importance.sort_values("importance", ascending=True).tail(10)
    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["importance"], color="#6a994e")
    plt.xlabel("Permutation importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_importance.png", dpi=180)
    plt.close()


def save_response_surface() -> None:
    dye = np.linspace(0.05, 8.0, 70)
    diameter = np.linspace(40, 2_000, 70)
    d_grid, s_grid = np.meshgrid(dye, diameter)
    area = np.pi * (s_grid * 1e-6) ** 2 * 80
    intensity = 0.38 * gaussian_absorption(550) * langmuir_saturation(d_grid) * area * 8.0e-2

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(d_grid, s_grid, intensity, cmap="viridis", linewidth=0, antialiased=True, alpha=0.94)
    ax.set_xlabel("Dye concentration (ug/mL)")
    ax.set_ylabel("Particle diameter (um)")
    ax.set_zlabel("Fluorescence intensity (W)")
    ax.set_title("3D Response Surface")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "response_surface_3d.png", dpi=180)
    plt.close()


def save_filtration_plot(filtration: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(filtration))
    ax1.bar(x - 0.18, filtration["removal_efficiency"], width=0.36, label="Efficiency", color="#0b6e69")
    ax1.bar(x + 0.18, filtration["fouling_alert_sensitivity"], width=0.36, label="Alert sensitivity", color="#f2a541")
    ax1.set_xticks(x)
    ax1.set_xticklabels(filtration["filter_system"], rotation=12, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Fraction")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(x, filtration["annual_cost_usd"], color="#c44536", marker="o", linewidth=2, label="Annual cost")
    ax2.set_ylabel("Annual cost (USD)")
    plt.title("Filtration and Fouling Comparison")
    fig.tight_layout()
    plt.savefig(FIGURE_DIR / "filtration_comparison.png", dpi=180)
    plt.close()
