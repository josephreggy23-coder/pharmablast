import numpy as np
import pandas as pd


def simulate_filtration(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    systems = {
        "single_stage_10um": {"efficiency": 0.78, "base_lifespan_days": 42, "cartridge_cost": 38, "service_cost": 22},
        "two_stage_50um_10um": {"efficiency": 0.91, "base_lifespan_days": 73, "cartridge_cost": 64, "service_cost": 28},
    }

    rows = []
    daily_load = rng.lognormal(mean=np.log(1.0), sigma=0.35, size=365)
    for name, cfg in systems.items():
        fouling = np.cumsum(daily_load * (1.0 - cfg["efficiency"] * 0.35))
        threshold = cfg["base_lifespan_days"] * np.mean(daily_load)
        alert_day = int(np.argmax(fouling > threshold)) if np.any(fouling > threshold) else 365
        lifespan = min(cfg["base_lifespan_days"] * rng.normal(1.0, 0.08), alert_day)
        replacements = int(np.ceil(365 / max(lifespan, 1)))
        annual_cost = replacements * (cfg["cartridge_cost"] + cfg["service_cost"])
        true_fouling_events = fouling > threshold
        alert_signal = fouling + rng.normal(0, threshold * 0.07, size=fouling.size)
        alerts = alert_signal > threshold
        sensitivity = np.sum(alerts & true_fouling_events) / max(np.sum(true_fouling_events), 1)
        rows.append(
            {
                "filter_system": name,
                "removal_efficiency": cfg["efficiency"],
                "estimated_lifespan_days": lifespan,
                "annual_replacements": replacements,
                "annual_cost_usd": annual_cost,
                "fouling_alert_sensitivity": sensitivity,
            }
        )

    return pd.DataFrame(rows)
