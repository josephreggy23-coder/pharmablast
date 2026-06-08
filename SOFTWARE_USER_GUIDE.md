# Software User Guide

## Starting the Software

Run the full build first:

```bash
python main.py
```

Then start the dashboard:

```bash
streamlit run app.py
```

## Model Simulator

Use the sliders to set polymer type, particle count, particle size, dye concentration, excitation wavelength, excitation intensity, pharmaceutical concentration, alpha modulation coefficient, and sensor noise level.

The output cards show predicted fluorescence intensity, signal-to-noise ratio, detection threshold status, saturation factor, wavelength efficiency, and drug modulation factor.

## Slider Meanings

- Polymer type: selects PS, PE, PP, or PET quantum yield.
- Particle count: number of particles contributing signal.
- Mean particle size: diameter used for surface-area scaling.
- Dye concentration: Nile Red concentration in ug/mL.
- Excitation wavelength: 450, 488, or 550 nm.
- Excitation intensity: relative light intensity.
- Pharmaceutical concentration: modeled contamination level.
- Alpha coefficient: negative values quench, positive values enhance.
- Sensor noise level: scales electronics noise.

## Interpreting Fluorescence

Higher modeled voltage means a stronger fluorescence signal at the photodiode. It depends on polymer brightness, dye saturation, particle area, wavelength selection, excitation intensity, and pharmaceutical modulation.

## Interpreting SNR

SNR compares ideal signal against estimated electronics noise. Higher SNR means easier detection. Low SNR near the threshold means the signal may be difficult to distinguish from sensor/background noise.

## Generating Modeling Data

Open the Synthetic Dataset Generator tab, select parameter ranges, choose a sample count and seed, then generate a CSV. The data are synthetic physics-model outputs, not ML training results.

## Interpreting 2D Plots

Use the 2D plots to isolate one effect at a time: dye saturation, polymer brightness, wavelength efficiency, particle size, pharmaceutical modulation, sensor threshold, condition improvement, and validation.

## Interpreting 3D Plots

Use the 3D plots to show interactions between parameters. The response surface is usually the strongest visual for judges because it shows why the useful dye range and particle size both matter.

## Exporting Files

Use the Export Center tab to access generated CSV files, reports, 2D images, and 3D HTML files. Running `python main.py` refreshes these outputs automatically.

## Troubleshooting

- If figures are missing, run `python main.py`.
- If the dashboard does not start, confirm Streamlit is installed with `pip install -r requirements.txt`.
- If SVG export is unavailable for a future Plotly figure, use the PNG and interactive HTML exports.
- If values seem too large or too small, check particle count, diameter, and excitation intensity first.
