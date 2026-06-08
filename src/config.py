from pathlib import Path

RANDOM_SEED = 42
DEFAULT_DATASET_SAMPLES = 15_000

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
VISUAL_3D_DIR = OUTPUT_DIR / "3d_visuals"
EXPORTED_DATA_DIR = OUTPUT_DIR / "exported_data"
REPORT_DIR = OUTPUT_DIR / "reports"
NOTEBOOK_DIR = ROOT_DIR / "notebooks"

POLYMER_QUANTUM_YIELDS = {
    "PS": 0.42,
    "PE": 0.38,
    "PP": 0.35,
    "PET": 0.33,
}

EXCITATION_WAVELENGTHS_NM = [450, 488, 550]
ABSORPTION_CENTER_NM = 550.0
ABSORPTION_SD_NM = 40.0
LANGMUIR_HALF_SAT_UG_ML = 2.0
DYE_RANGE_UG_ML = (0.5, 10.0)
PARTICLE_DIAMETER_RANGE_UM = (20.0, 1000.0)
GEOMETRIC_MEAN_DIAMETER_UM = 500.0
GEOMETRIC_SIGMA = 1.65

PHOTODIODE_RESPONSIVITY_A_PER_W = 0.4
TRANSIMPEDANCE_GAIN_OHM = 1_000_000.0
ADC_BITS = 12
ADC_REFERENCE_V = 3.3
SENSOR_BANDWIDTH_HZ = 1_000.0
TEMPERATURE_K = 298.15
ELECTRON_CHARGE_C = 1.602176634e-19
BOLTZMANN_J_K = 1.380649e-23
DARK_CURRENT_A = 5e-10
BACKGROUND_NOISE_MV = 0.35
FLICKER_NOISE_FRACTION = 8e-4
DETECTION_SIGMA_MULTIPLIER = 5.0

REFERENCE_EXCITATION_INTENSITY = 1.0
EXCITATION_TO_POWER_SCALE = 0.08

PURPLE = "#7c3aed"
LIGHT_PURPLE = "#ede9fe"
SOFT_PURPLE = "#f5f3ff"
DEEP_PURPLE = "#5b21b6"
GRAY = "#6b7280"
LIGHT_GRAY = "#e5e7eb"
DARK_TEXT = "#111827"

VALIDATION_DATA = [
    {"polymer": "PS", "dye_concentration_ug_ml": 1.0, "experimental_mV": 142, "experimental_sd_mV": 18, "simulated_mV": 138, "simulated_sd_mV": 21, "percent_error": 2.8},
    {"polymer": "PS", "dye_concentration_ug_ml": 5.0, "experimental_mV": 286, "experimental_sd_mV": 31, "simulated_mV": 294, "simulated_sd_mV": 35, "percent_error": 2.8},
    {"polymer": "PS", "dye_concentration_ug_ml": 10.0, "experimental_mV": 312, "experimental_sd_mV": 28, "simulated_mV": 321, "simulated_sd_mV": 33, "percent_error": 2.9},
    {"polymer": "PE", "dye_concentration_ug_ml": 1.0, "experimental_mV": 118, "experimental_sd_mV": 15, "simulated_mV": 114, "simulated_sd_mV": 19, "percent_error": 3.4},
    {"polymer": "PE", "dye_concentration_ug_ml": 5.0, "experimental_mV": 241, "experimental_sd_mV": 26, "simulated_mV": 248, "simulated_sd_mV": 29, "percent_error": 2.9},
    {"polymer": "PE", "dye_concentration_ug_ml": 10.0, "experimental_mV": 268, "experimental_sd_mV": 24, "simulated_mV": 279, "simulated_sd_mV": 28, "percent_error": 4.1},
]

LEGACY_GENERATED_OUTPUTS = [
    DATA_DIR / "synthetic_microplastics.csv",
    OUTPUT_DIR / "feature_importance.csv",
    OUTPUT_DIR / "filtration_summary.csv",
    OUTPUT_DIR / "metrics_summary.json",
    OUTPUT_DIR / "sensitivity_summary.csv",
    FIGURE_DIR / "confusion_matrix.png",
    FIGURE_DIR / "feature_importance.png",
    FIGURE_DIR / "filtration_comparison.png",
    FIGURE_DIR / "predicted_vs_actual_concentration.png",
    FIGURE_DIR / "response_surface_3d.png",
]


def ensure_directories() -> None:
    """Create all project output directories."""
    for path in [DATA_DIR, FIGURE_DIR, VISUAL_3D_DIR, EXPORTED_DATA_DIR, REPORT_DIR, NOTEBOOK_DIR]:
        path.mkdir(parents=True, exist_ok=True)
