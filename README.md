# Synthetic Data Engine for Fluorescence-Based Detection of Pharmaceutical-Laden Microplastics in Water Systems

This repository contains physics-informed modeling software for exploring Nile Red fluorescence detection of pharmaceutical-laden microplastics in water systems. It is designed as a competition-ready computational modeling platform: users can vary polymer type, dye concentration, excitation wavelength, particle size, pharmaceutical modulation, and sensor noise, then export clean 2D and 3D results.

The software is modeling-first. It does not train machine learning classifiers, predict polymer identity, or claim deployment-ready performance.

## Why This Matters

Microplastic detection in water is difficult because signal strength depends on particle material, particle size, dye binding, excitation wavelength, optical collection, sensor noise, and possible chemical interference. A physics-informed software model helps explore detection conditions before building every physical version. Synthetic modeling data can guide experiments, but it does not replace real-world validation.

## Main Physics Equation

```text
I = Qp * A(lambda) * S(cd) * As * E * Mdrug
```

Where:

- `Qp` is polymer-specific quantum yield.
- `A(lambda)` is Nile Red wavelength absorption efficiency, modeled as a Gaussian centered at 550 nm with SD 40 nm.
- `S(cd)` is Langmuir dye saturation with half-saturation near 2.0 ug/mL.
- `As` is particle surface area from particle diameter and count.
- `E` is excitation intensity.
- `Mdrug = 1 + alpha * Cdrug` is pharmaceutical fluorescence modulation.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Modeling Build

```bash
python main.py
```

This creates:

- `data/synthetic_modeling_data.csv`
- `data/validation_data.csv`
- `outputs/figures/`
- `outputs/3d_visuals/`
- `outputs/exported_data/`
- `outputs/reports/software_summary.md`

## Run the Dashboard

```bash
streamlit run app.py
```

The dashboard tabs are:

1. Model Simulator
2. 2D Model Visualizations
3. 3D Model Explorer
4. Synthetic Dataset Generator
5. Validation and Assumptions
6. Export Center
7. Quick Demo

## Output Folders

- `data/`: generated modeling dataset and validation table.
- `outputs/figures/`: 2D PNG/SVG figures.
- `outputs/3d_visuals/`: interactive HTML 3D models plus static PNG/SVG where supported.
- `outputs/exported_data/`: files prepared for download or sharing.
- `outputs/reports/`: build summary and software summary report.

## Scientific Scope

The model includes:

- PS, PE, PP, and PET polymer classes.
- Quantum yields: PS 0.42, PE 0.38, PP 0.35, PET 0.33.
- Langmuir dye saturation from 0.5 to 10.0 ug/mL.
- Gaussian wavelength efficiency centered at 550 nm.
- Log-normal particle size statistics around a 500 um geometric mean.
- Silicon photodiode responsivity around 0.4 A/W at 630 nm.
- Transimpedance amplifier, 12-bit ADC, shot noise, Johnson noise, flicker noise, dark current/background noise, SNR, and detection threshold.
- Controlled validation comparison for PS/PE fluorescence values.

## Limitations

This is a physics-informed modeling platform, not a finished field instrument. It does not prove performance in every environmental condition, and it is not ready for real wastewater deployment. The model is grounded in fluorescence physics and compared against controlled validation data, but future work requires broader testing with real environmental samples, more pharmaceuticals, mixed polymers, turbidity, biofouling, and instrument calibration data.

## Future Improvements

- Add experimentally measured alpha coefficients for specific pharmaceuticals.
- Compare against larger laboratory validation datasets.
- Add mixture modeling for multiple polymer types in one sample.
- Add optical path length and filter transmission calibration when measured data are available.
- Connect the dashboard to laboratory data files for side-by-side model refinement.
