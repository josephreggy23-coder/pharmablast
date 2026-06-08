import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


CLASSIFICATION_FEATURES = [
    "particle_count",
    "dye_concentration_ug_ml",
    "excitation_wavelength_nm",
    "particle_mean_diameter_um",
    "particle_std_diameter_um",
    "fluorescence_intensity_w",
    "estimated_quantum_yield",
    "signal_to_noise_ratio",
    "saturation_fraction",
    "drug_modulation",
    "drug_alpha",
    "filtration_state",
]

REGRESSION_FEATURES = [
    "polymer_type",
    "quantum_yield",
    "particle_count",
    "dye_concentration_ug_ml",
    "excitation_wavelength_nm",
    "particle_mean_diameter_um",
    "particle_std_diameter_um",
    "fluorescence_intensity_w",
    "signal_to_noise_ratio",
    "saturation_fraction",
    "drug_modulation",
    "filtration_state",
]


def _preprocessor(features: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric = [f for f in features if f not in categorical]
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def train_classifiers(df: pd.DataFrame, seed: int) -> dict:
    x = df[CLASSIFICATION_FEATURES]
    y = df["polymer_type"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=seed)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=260, max_depth=12, min_samples_leaf=3, random_state=seed, n_jobs=-1),
        "SVM": SVC(C=4.0, gamma="scale", kernel="rbf", random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=180, learning_rate=0.045, max_depth=3, random_state=seed),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", _preprocessor(CLASSIFICATION_FEATURES, ["filtration_state"])), ("model", model)])
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        results[name] = {
            "pipeline": pipe,
            "accuracy": float(accuracy_score(y_test, pred)),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(y.unique())),
            "labels": sorted(y.unique()),
            "x_test": x_test,
            "y_test": y_test,
            "pred": pred,
        }

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    results["best_name"] = best_name
    return results


def train_concentration_regressor(df: pd.DataFrame, seed: int) -> dict:
    x = df[REGRESSION_FEATURES]
    y = df["drug_concentration_ug_l"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed)

    model = RandomForestRegressor(n_estimators=320, max_depth=18, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    pipe = Pipeline([("prep", _preprocessor(REGRESSION_FEATURES, ["polymer_type", "filtration_state"])), ("model", model)])
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    return {
        "pipeline": pipe,
        "x_test": x_test,
        "y_test": y_test,
        "pred": pred,
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }


def feature_importance(regression_result: dict, seed: int) -> pd.DataFrame:
    importance = permutation_importance(
        regression_result["pipeline"],
        regression_result["x_test"],
        regression_result["y_test"],
        n_repeats=8,
        random_state=seed,
        n_jobs=-1,
    )
    out = pd.DataFrame({"feature": REGRESSION_FEATURES, "importance": importance.importances_mean})
    out["importance_fraction"] = out["importance"] / out["importance"].clip(lower=0).sum()
    return out.sort_values("importance", ascending=False)
