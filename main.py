import argparse
import json
from pathlib import Path

from src.config import (
    DATA_DIR,
    DEFAULT_DATASET_SAMPLES,
    EXPORTED_DATA_DIR,
    RANDOM_SEED,
    ROOT_DIR,
    ensure_directories,
)
from src.synthetic_dataset import generate_synthetic_modeling_data
from src.utils import average_validation_error, cleanup_legacy_outputs, copy_for_export, save_validation_data, write_software_summary
from src.visualization_2d import generate_all_2d_figures
from src.visualization_3d import generate_all_3d_visuals


def build_project(n_samples: int = DEFAULT_DATASET_SAMPLES, seed: int = RANDOM_SEED) -> dict:
    """Generate all modeling data, figures, 3D visuals, and reports."""
    ensure_directories()
    cleanup_legacy_outputs()

    dataset = generate_synthetic_modeling_data(n_samples=n_samples, seed=seed)
    dataset_path = DATA_DIR / "synthetic_modeling_data.csv"
    dataset.to_csv(dataset_path, index=False)
    copy_for_export(dataset_path)

    validation = save_validation_data()
    avg_error = average_validation_error()

    figure_paths = generate_all_2d_figures()
    visual_paths = generate_all_3d_visuals()
    report_path = write_software_summary(len(dataset), avg_error, len([p for p in figure_paths if p.endswith(".png")]), len(visual_paths))

    summary = {
        "project": "Synthetic Data Engine for Fluorescence-Based Detection of Pharmaceutical-Laden Microplastics in Water Systems",
        "dataset": str(dataset_path),
        "dataset_rows": int(len(dataset)),
        "validation_data": str(DATA_DIR / "validation_data.csv"),
        "average_validation_error_percent": avg_error,
        "figures_created": figure_paths,
        "three_d_visuals_created": visual_paths,
        "software_summary": str(report_path),
        "exported_data_folder": str(EXPORTED_DATA_DIR),
        "validation_rows": int(len(validation)),
        "scope": "Physics-informed modeling and visualization; no ML workflow is used.",
    }
    summary_path = report_path.parent / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the microplastic fluorescence modeling platform outputs.")
    parser.add_argument("--samples", type=int, default=DEFAULT_DATASET_SAMPLES, help="Number of synthetic modeling rows to generate.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducible outputs.")
    args = parser.parse_args()

    summary = build_project(n_samples=args.samples, seed=args.seed)
    print("\nPhysics-based modeling build complete.")
    print(f"Dataset: {summary['dataset']}")
    print(f"Rows: {summary['dataset_rows']:,}")
    print(f"Average validation error: {summary['average_validation_error_percent']:.2f}%")
    print(f"2D figure files: {len(summary['figures_created'])}")
    print(f"3D visual files: {len(summary['three_d_visuals_created'])}")
    print(f"Report: {summary['software_summary']}")
    print(f"Project root: {ROOT_DIR}")


if __name__ == "__main__":
    main()
