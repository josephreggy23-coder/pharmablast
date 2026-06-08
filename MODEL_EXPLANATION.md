# Model Explanation

## Main Equation

```text
I = Qp * A(lambda) * S(cd) * As * E * Mdrug
```

This equation predicts fluorescence intensity from polymer brightness, wavelength efficiency, dye saturation, particle surface area, excitation intensity, and pharmaceutical modulation.

## Terms

- `I`: predicted fluorescence intensity.
- `Qp`: polymer-specific quantum yield.
- `A(lambda)`: wavelength-dependent Nile Red absorption efficiency.
- `S(cd)`: dye saturation as concentration increases.
- `As`: particle surface area.
- `E`: excitation light intensity.
- `Mdrug`: pharmaceutical modulation factor.

## Langmuir Saturation

The model uses Langmuir saturation:

```text
S(cd) = cd / (Kd + cd)
```

`Kd` is set near 2.0 ug/mL. Fluorescence rises quickly at low dye concentration, then begins to saturate. The dashboard marks the useful operating range around 3 to 5 ug/mL and the saturation zone above about 5 ug/mL.

## Particle Size and Surface Area

Particles are approximated as spheres. Surface area scales with diameter squared:

```text
As = particle_count * pi * diameter^2
```

Because fluorescence is modeled as surface-associated Nile Red signal, larger particles produce stronger signal when all other factors are fixed.

## Quantum Yield and Polymer Brightness

The model uses the following quantum yields:

- PS = 0.42
- PE = 0.38
- PP = 0.35
- PET = 0.33

Higher quantum yield means a brighter modeled fluorescence response under the same illumination, dye concentration, and particle size.

## Pharmaceutical Modulation

Pharmaceutical effects are modeled as:

```text
Mdrug = 1 + alpha * Cdrug
```

Negative alpha values represent quenching examples. Positive alpha values represent enhancement examples. This is a modeled effect and should be refined with real chemical measurements for specific pharmaceuticals.

## Wavelength Efficiency

Nile Red wavelength efficiency is modeled as a Gaussian centered at 550 nm with standard deviation 40 nm. The software marks 450, 488, and 550 nm so users can see why excitation wavelength matters.

## Photodiode Noise

The sensor model includes:

- silicon photodiode responsivity around 0.4 A/W at 630 nm
- transimpedance amplifier conversion to voltage
- 12-bit ADC quantization
- shot noise
- Johnson noise
- flicker noise
- dark current/background noise
- signal-to-noise ratio
- detection threshold

## Synthetic Modeling Data

The generated CSV is synthetic modeling data. It is produced from the physics equations and fixed random seeds. It helps explore expected behavior across parameter ranges, but it is not a substitute for laboratory measurements.

## Why Validation Is Still Needed

The validation table compares controlled PS/PE fluorescence values with simulated values and gives an average percent error near 3.2%. That controlled comparison supports the modeling approach, but it does not prove performance in real water systems. Future validation should include environmental samples, turbidity, mixed plastics, pharmaceutical mixtures, and repeated instrument calibration.

## 2D Visualizations

- Langmuir Saturation Curve: shows dye concentration response and saturation behavior.
- Polymer Fluorescence Comparison: compares PS, PE, PP, and PET under the same condition.
- Wavelength Efficiency Curve: shows the Gaussian absorption model and marked excitation wavelengths.
- Particle Size Effect: shows nonlinear fluorescence growth from surface-area scaling.
- Pharmaceutical Modulation: shows modeled quenching and enhancement.
- Sensor Noise and Detection Threshold: shows signal, noise envelope, and low-signal detection difficulty.
- Baseline vs Improved Condition: compares one baseline condition with a reasonable improved condition.
- Validation Plot: compares supplied controlled experimental and simulated values.

## 3D Visualizations

- 3D Response Surface: dye concentration and particle diameter jointly control fluorescence.
- 3D Parameter Space Explorer: dye concentration, excitation wavelength, and particle size interact.
- 3D Pharmaceutical Effect Surface: pharmaceutical concentration and alpha affect quenching/enhancement.
- 3D Sensor Chamber Model: visual engineering model of chamber, particles, excitation beam, emission, filter, photodiode, and signal processing.
