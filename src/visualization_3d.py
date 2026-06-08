import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .config import DEEP_PURPLE, LIGHT_GRAY, LIGHT_PURPLE, PURPLE, SOFT_PURPLE, VISUAL_3D_DIR
from .physics_model import drug_modulation, predict_fluorescence_power_w
from .sensor_model import sensor_response


def _mV(polymer="PE", count=80, diameter=500, dye=4, wavelength=550, pharma=0, alpha=0, excitation=1.0):
    power = predict_fluorescence_power_w(polymer, count, diameter, dye, wavelength, excitation, pharma, alpha)
    return float(sensor_response(power, include_random=False)["ideal_voltage_mV"])


def _plotly_layout(title: str) -> dict:
    return {
        "title": {"text": title, "x": 0.02},
        "template": "plotly_white",
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": "Arial, sans-serif", "color": "#111827", "size": 13},
        "legend": {"bgcolor": "rgba(255,255,255,0.85)", "bordercolor": LIGHT_GRAY, "borderwidth": 1},
        "margin": {"l": 0, "r": 0, "t": 60, "b": 0},
    }


def _write_html(fig: go.Figure, name: str) -> str:
    VISUAL_3D_DIR.mkdir(parents=True, exist_ok=True)
    path = VISUAL_3D_DIR / f"{name}.html"
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    return str(path)


def response_surface_arrays():
    dye = np.linspace(0.5, 10.0, 80)
    diameter = np.linspace(20, 1000, 80)
    dye_grid, diameter_grid = np.meshgrid(dye, diameter)
    z = np.zeros_like(dye_grid)
    for i in range(diameter_grid.shape[0]):
        for j in range(dye_grid.shape[1]):
            z[i, j] = _mV(diameter=float(diameter_grid[i, j]), dye=float(dye_grid[i, j]), wavelength=550)
    return dye_grid, diameter_grid, z


def create_response_surface() -> list[str]:
    dye_grid, diameter_grid, z = response_surface_arrays()
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=dye_grid,
            y=diameter_grid,
            z=z,
            colorscale=[[0, "#f5f3ff"], [0.45, "#c4b5fd"], [1, "#6d28d9"]],
            colorbar={"title": "mV"},
            name="Fluorescence",
        )
    )
    for dye_value, label in [(3.0, "useful range begins"), (5.0, "saturation zone begins")]:
        diam = np.linspace(20, 1000, 80)
        signal = [_mV(diameter=float(d), dye=dye_value, wavelength=550) for d in diam]
        fig.add_trace(go.Scatter3d(x=[dye_value] * len(diam), y=diam, z=signal, mode="lines", line={"color": DEEP_PURPLE, "width": 6}, name=label))
    fig.update_layout(
        **_plotly_layout("Computational Optimization of Fluorescence Detection"),
        scene={
            "xaxis_title": "Nile Red concentration (ug/mL)",
            "yaxis_title": "Particle diameter (um)",
            "zaxis_title": "Predicted fluorescence (mV)",
            "xaxis": {"backgroundcolor": "white", "gridcolor": LIGHT_GRAY},
            "yaxis": {"backgroundcolor": "white", "gridcolor": LIGHT_GRAY},
            "zaxis": {"backgroundcolor": "white", "gridcolor": LIGHT_GRAY},
        },
    )
    paths = [_write_html(fig, "response_surface")]

    static = plt.figure(figsize=(9, 7))
    ax = static.add_subplot(111, projection="3d")
    ax.plot_surface(dye_grid, diameter_grid, z, cmap="Purples", linewidth=0, antialiased=True, alpha=0.96)
    ax.set_title("Computational Optimization of Fluorescence Detection", fontsize=14, weight="bold", pad=16)
    ax.set_xlabel("Nile Red concentration (ug/mL)", labelpad=10)
    ax.set_ylabel("Particle diameter (um)", labelpad=10)
    ax.set_zlabel("Predicted fluorescence (mV)", labelpad=10)
    ax.set_facecolor("white")
    static.patch.set_facecolor("white")
    ax.xaxis.pane.set_facecolor((1, 1, 1, 1))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 1))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 1))
    png = VISUAL_3D_DIR / "response_surface.png"
    svg = VISUAL_3D_DIR / "response_surface.svg"
    static.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    static.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(static)
    paths.extend([str(png), str(svg)])
    return paths


def create_parameter_space_explorer() -> str:
    rng = np.random.default_rng(42)
    dye = rng.uniform(0.5, 10.0, 520)
    wavelength = rng.uniform(430, 590, 520)
    diameter = rng.uniform(20, 1000, 520)
    signal = np.array([_mV(diameter=d, dye=c, wavelength=w) for d, c, w in zip(diameter, dye, wavelength)])
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=dye,
                y=wavelength,
                z=signal,
                mode="markers",
                marker={"size": 4 + 5 * (diameter - diameter.min()) / (diameter.max() - diameter.min()), "color": diameter, "colorscale": "Purples", "opacity": 0.82, "colorbar": {"title": "Diameter (um)"}},
                name="Modeled condition",
            )
        ]
    )
    fig.update_layout(
        **_plotly_layout("3D Parameter Space Explorer"),
        scene={
            "xaxis_title": "Dye concentration (ug/mL)",
            "yaxis_title": "Excitation wavelength (nm)",
            "zaxis_title": "Predicted fluorescence (mV)",
        },
    )
    return _write_html(fig, "parameter_space_explorer")


def create_pharmaceutical_effect_surface() -> str:
    concentration = np.linspace(0, 10, 80)
    alpha = np.linspace(-0.08, 0.08, 80)
    c_grid, a_grid = np.meshgrid(concentration, alpha)
    base = _mV(diameter=500, dye=4, wavelength=550, pharma=0, alpha=0)
    z = base * drug_modulation(c_grid, a_grid)
    fig = go.Figure(
        data=[
            go.Surface(
                x=c_grid,
                y=a_grid,
                z=z,
                colorscale=[[0, "#ede9fe"], [0.5, "#a78bfa"], [1, "#5b21b6"]],
                colorbar={"title": "mV"},
            )
        ]
    )
    fig.add_trace(go.Scatter3d(x=[0, 10], y=[0, 0], z=[base, base], mode="lines", line={"color": "#ef4444", "width": 6}, name="No modulation boundary"))
    fig.update_layout(
        **_plotly_layout("3D Pharmaceutical Effect Surface"),
        scene={
            "xaxis_title": "Pharmaceutical concentration (model units)",
            "yaxis_title": "Alpha modulation coefficient",
            "zaxis_title": "Predicted fluorescence (mV)",
        },
    )
    return _write_html(fig, "pharmaceutical_effect_surface")


def _box_vertices(x0, x1, y0, y1, z0, z1):
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )


def _add_plotly_box(fig, name, bounds, color, opacity=0.25):
    vertices = _box_vertices(*bounds)
    triangles = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (2, 3, 7),
        (2, 7, 6),
        (1, 2, 6),
        (1, 6, 5),
        (0, 3, 7),
        (0, 7, 4),
    ]
    i, j, k = zip(*triangles)
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=opacity,
            name=name,
            showscale=False,
        )
    )


def create_sensor_chamber() -> list[str]:
    rng = np.random.default_rng(7)
    fig = go.Figure()
    _add_plotly_box(fig, "Transparent water sample chamber", (0, 4, 0, 2.5, 0, 2.5), LIGHT_PURPLE, 0.18)
    _add_plotly_box(fig, "Optical filter", (4.25, 4.45, 0.35, 2.15, 0.35, 2.15), "#c4b5fd", 0.55)
    _add_plotly_box(fig, "Photodiode sensor", (4.7, 5.2, 0.6, 1.9, 0.6, 1.9), "#ddd6fe", 0.8)
    _add_plotly_box(fig, "ADC / Signal Processing", (5.55, 6.8, 0.45, 2.05, 0.45, 2.05), "#f5f3ff", 0.9)

    particles = rng.uniform([0.45, 0.35, 0.35], [3.55, 2.15, 2.15], size=(44, 3))
    fig.add_trace(go.Scatter3d(x=particles[:, 0], y=particles[:, 1], z=particles[:, 2], mode="markers", marker={"size": 5, "color": "#fb923c", "opacity": 0.82}, name="Fluorescent particles"))
    fig.add_trace(go.Scatter3d(x=[-0.8, 0.2, 1.2, 2.2], y=[1.25] * 4, z=[1.25] * 4, mode="lines", line={"color": "#2563eb", "width": 8}, name="Excitation beam"))
    for p in particles[::7]:
        fig.add_trace(go.Scatter3d(x=[p[0], 4.25], y=[p[1], 1.25], z=[p[2], 1.25], mode="lines", line={"color": "#f97316", "width": 3}, showlegend=False))
    fig.add_trace(go.Scatter3d(x=[4.45, 4.7, 5.2, 5.55], y=[1.25] * 4, z=[1.25] * 4, mode="lines", line={"color": PURPLE, "width": 5}, name="Signal flow"))
    labels = [
        ("Water sample chamber", 2.0, 2.8, 2.65),
        ("Optical filter", 4.35, 2.45, 2.1),
        ("Photodiode", 4.95, 2.25, 2.05),
        ("ADC / Signal Processing", 6.18, 2.35, 2.1),
        ("Blue excitation", -0.35, 1.55, 1.55),
        ("Orange/red emission", 2.9, 1.8, 1.7),
    ]
    fig.add_trace(go.Scatter3d(x=[x for _, x, _, _ in labels], y=[y for _, _, y, _ in labels], z=[z for _, _, _, z in labels], mode="text", text=[t for t, _, _, _ in labels], textfont={"size": 12, "color": "#111827"}, showlegend=False))
    fig.update_layout(
        **_plotly_layout("3D Sensor Chamber Model"),
        scene={"xaxis": {"visible": False}, "yaxis": {"visible": False}, "zaxis": {"visible": False}, "aspectmode": "data"},
    )
    paths = [_write_html(fig, "sensor_chamber")]

    static = plt.figure(figsize=(10, 6))
    ax = static.add_subplot(111, projection="3d")
    _draw_box(ax, (0, 4, 0, 2.5, 0, 2.5), LIGHT_PURPLE, 0.18)
    _draw_box(ax, (4.25, 4.45, 0.35, 2.15, 0.35, 2.15), "#c4b5fd", 0.55)
    _draw_box(ax, (4.7, 5.2, 0.6, 1.9, 0.6, 1.9), "#ddd6fe", 0.8)
    _draw_box(ax, (5.55, 6.8, 0.45, 2.05, 0.45, 2.05), "#f5f3ff", 0.9)
    ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], color="#fb923c", s=18, alpha=0.85)
    ax.plot([-0.8, 3.6], [1.25, 1.25], [1.25, 1.25], color="#2563eb", linewidth=4)
    for p in particles[::6]:
        ax.plot([p[0], 4.25], [p[1], 1.25], [p[2], 1.25], color="#f97316", linewidth=1.4, alpha=0.75)
    ax.plot([4.45, 6.8], [1.25, 1.25], [1.25, 1.25], color=PURPLE, linewidth=3)
    for text, x, y, z in labels:
        ax.text(x, y, z, text, fontsize=9, color="#111827")
    ax.set_title("3D Sensor Chamber Model", fontsize=14, weight="bold")
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-62)
    static.patch.set_facecolor("white")
    png = VISUAL_3D_DIR / "sensor_chamber.png"
    svg = VISUAL_3D_DIR / "sensor_chamber.svg"
    static.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    static.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(static)
    paths.extend([str(png), str(svg)])
    return paths


def _draw_box(ax, bounds, color, alpha):
    x0, x1, y0, y1, z0, z1 = bounds
    v = _box_vertices(x0, x1, y0, y1, z0, z1)
    faces = [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[0], v[3], v[7], v[4]],
    ]
    poly = Poly3DCollection(faces, facecolors=color, edgecolors=DEEP_PURPLE, linewidths=0.6, alpha=alpha)
    ax.add_collection3d(poly)


def generate_all_3d_visuals() -> list[str]:
    """Generate all required interactive 3D files and static exports where supported."""
    VISUAL_3D_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.extend(create_response_surface())
    paths.append(create_parameter_space_explorer())
    paths.append(create_pharmaceutical_effect_surface())
    paths.extend(create_sensor_chamber())
    return paths
