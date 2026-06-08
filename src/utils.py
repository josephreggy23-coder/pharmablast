import shutil
from pathlib import Path

import pandas as pd

from .config import (
    DATA_DIR,
    EXPORTED_DATA_DIR,
    LEGACY_GENERATED_OUTPUTS,
    REPORT_DIR,
    VALIDATION_DATA,
    ensure_directories,
)


def cleanup_legacy_outputs() -> None:
    """Remove generated ML-era artifacts so the current workflow stays modeling-first."""
    for path in LEGACY_GENERATED_OUTPUTS:
        if path.exists() and path.is_file():
            path.unlink()


def validation_dataframe() -> pd.DataFrame:
    """Return the controlled validation values supplied for the project."""
    df = pd.DataFrame(VALIDATION_DATA)
    df["sample"] = df["polymer"] + " " + df["dye_concentration_ug_ml"].astype(str) + " ug/mL"
    return df


def save_validation_data() -> pd.DataFrame:
    """Save validation values to data and export folders."""
    ensure_directories()
    df = validation_dataframe()
    df.to_csv(DATA_DIR / "validation_data.csv", index=False)
    df.to_csv(EXPORTED_DATA_DIR / "validation_data.csv", index=False)
    return df


def average_validation_error() -> float:
    """Average percent error for supplied PS/PE validation cases."""
    return float(validation_dataframe()["percent_error"].mean())


def copy_for_export(source: Path, destination_name: str | None = None) -> Path:
    """Copy an output file to the export folder for easy download."""
    ensure_directories()
    destination = EXPORTED_DATA_DIR / (destination_name or source.name)
    shutil.copy2(source, destination)
    return destination


def write_software_summary(dataset_rows: int, avg_error: float, figure_count: int, visual_count: int) -> Path:
    """Create a short report that can be shared with judges."""
    ensure_directories()
    path = REPORT_DIR / "software_summary.md"
    path.write_text(
        "\n".join(
            [
                "# Software Summary",
                "",
                "Project: Synthetic Data Engine for Fluorescence-Based Detection of Pharmaceutical-Laden Microplastics in Water Systems",
                "",
                "This is a physics-informed modeling platform for exploring Nile Red fluorescence detection conditions.",
                "It does not replace laboratory testing or real environmental validation.",
                "",
                f"- Synthetic modeling rows generated: {dataset_rows:,}",
                f"- Average validation percent error: {avg_error:.2f}%",
                f"- 2D figures generated: {figure_count}",
                f"- 3D files generated: {visual_count}",
                "",
                "Main equation:",
                "",
                "`I = Qp * A(lambda) * S(cd) * As * E * Mdrug`",
                "",
                "Recommended judge demo order: equation, baseline condition, dye saturation, particle size effect, pharmaceutical modulation, 3D response surface, sensor chamber, export center.",
            ]
        ),
        encoding="utf-8",
    )
    return path
