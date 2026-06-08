# Judge Demo Script

## 30-Second Pitch

This is a physics-informed modeling platform for Nile Red fluorescence detection of pharmaceutical-laden microplastics. It lets me explore how polymer type, dye concentration, wavelength, particle size, pharmaceutical modulation, and sensor noise affect detection before building every physical version. The model is grounded in fluorescence physics and compared against controlled PS/PE validation data.

## 90-Second Pitch

The core model is `I = Qp * A(lambda) * S(cd) * As * E * Mdrug`. I use polymer-specific quantum yields, Langmuir dye saturation, a Gaussian wavelength response centered at 550 nm, particle surface-area scaling, pharmaceutical quenching or enhancement, and a photodiode sensor model with noise and ADC behavior. The dashboard lets a user change detection conditions interactively, view 2D plots, explore 3D parameter spaces, generate synthetic modeling data, and export figures and reports. It is not a replacement for laboratory testing, but it helps guide which experimental conditions are worth testing first.

## 3-Minute Technical Walkthrough

1. Start with the main equation and show each physical term.
2. Open the Model Simulator and show a baseline PE condition.
3. Increase dye concentration into the 3 to 5 ug/mL useful range and show the signal increase.
4. Increase dye above 5 ug/mL and explain saturation.
5. Increase particle size and explain surface-area scaling.
6. Move alpha negative and positive to show modeled quenching/enhancement.
7. Open the 3D Response Surface and point out the joint effect of dye and particle size.
8. Open the Sensor Chamber Model and explain the modeled optical/electronics path.
9. Open the Export Center and show the generated CSV, figures, HTML files, and report.

## Five Hard Judge Questions and Ideal Answers

**Question 1: Does this replace lab testing?**  
No. This is a physics-informed modeling platform. It helps explore detection conditions before building every physical version, but future work requires broader testing with real environmental samples.

**Question 2: Why use Langmuir saturation?**  
Langmuir saturation is a simple physically interpretable way to model dye binding: signal rises quickly at low concentration and then approaches a plateau as available binding/signal contribution saturates.

**Question 3: Why does particle size matter so much?**  
The model treats fluorescence as surface-associated signal, so exposed area scales approximately with diameter squared for spherical particles. That makes particle diameter a major signal driver.

**Question 4: What does pharmaceutical alpha mean?**  
Alpha is a modulation coefficient in `Mdrug = 1 + alpha * Cdrug`. Negative alpha represents a quenching example and positive alpha represents enhancement. Specific alpha values should be measured experimentally for specific pharmaceuticals.

**Question 5: What is the biggest limitation?**  
The biggest limitation is that synthetic modeling data do not capture every real water condition. Turbidity, biofouling, mixed polymers, environmental contaminants, and instrument calibration all require broader validation.

## Weaknesses and Honest Defense

This model simplifies particle shape, assumes a controlled optical setup, and uses a general pharmaceutical modulation term. Those choices make the software understandable and useful for experiment planning, but they do not prove field performance. The honest value is that the platform shows which variables matter and helps design better lab validation.

## Ethical AI Explanation

“Codex and Claude were used as coding and writing assistants. I designed the scientific model, selected the variables, reviewed the assumptions, interpreted the outputs, and verified that the model stayed consistent with the project paper. AI tools did not replace scientific reasoning or validation.”
