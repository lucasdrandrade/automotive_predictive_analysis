import pandas as pd

COLUMN_MAP = {
    "Engine rpm":       "engine_rpm",
    "Lub oil pressure": "lub_oil_pressure",
    "Fuel pressure":    "fuel_pressure",
    "Coolant pressure": "coolant_pressure",
    "lub oil temp":     "lub_oil_temp",
    "Coolant temp":     "coolant_temp",
    "Engine Condition": "engine_condition",
}

SENSOR_COLS = [
    "engine_rpm",
    "lub_oil_pressure",
    "fuel_pressure",
    "coolant_pressure",
    "lub_oil_temp",
    "coolant_temp",
]

SENSOR_LABELS = {
    "engine_rpm":       "Engine RPM",
    "lub_oil_pressure": "Lub Oil Pressure",
    "fuel_pressure":    "Fuel Pressure",
    "coolant_pressure": "Coolant Pressure",
    "lub_oil_temp":     "Lub Oil Temp (°C)",
    "coolant_temp":     "Coolant Temp (°C)",
}

DERIVED_COLS = ["oil_efficiency", "coolant_efficiency"]

DERIVED_LABELS = {
    "oil_efficiency":     "Oil Efficiency",
    "coolant_efficiency": "Coolant Efficiency",
}

ALL_FEATURE_COLS = SENSOR_COLS + DERIVED_COLS

ALL_FEATURE_LABELS = {**SENSOR_LABELS, **DERIVED_LABELS}

CONDITION_LABELS = {1: "Healthy", 0: "Faulty"}
CONDITION_COLORS = {1: "#27ae60", 0: "#e74c3c"}


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["oil_efficiency"] = 1.0 / (df["engine_rpm"] * df["lub_oil_temp"])
    df["coolant_efficiency"] = df["coolant_temp"] / df["engine_rpm"]
    return df


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_MAP)
    df = df.reset_index(drop=True)
    return add_derived_features(df)


def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    if total == 0:
        return {"total": 0, "healthy": 0, "faulty": 0, "fault_rate": 0.0, "healthy_rate": 0.0}
    healthy = int((df["engine_condition"] == 1).sum())
    faulty = int((df["engine_condition"] == 0).sum())
    return {
        "total": total,
        "healthy": healthy,
        "faulty": faulty,
        "fault_rate": round(faulty / total * 100, 1),
        "healthy_rate": round(healthy / total * 100, 1),
    }


def compute_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = ALL_FEATURE_COLS + ["engine_condition"]
    corr = df[cols].corr()["engine_condition"].drop("engine_condition").fillna(0)
    result = pd.DataFrame({
        "feature":     corr.index.tolist(),
        "label":       [ALL_FEATURE_LABELS.get(c, c) for c in corr.index],
        "correlation": corr.values,
        "is_derived":  [c in DERIVED_COLS for c in corr.index],
    })
    result["abs_corr"] = result["correlation"].abs()
    return result.sort_values("abs_corr", ascending=True).reset_index(drop=True)


def compute_regression_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    if len(df) < 50 or df["engine_condition"].nunique() < 2:
        return pd.DataFrame(columns=["feature", "label", "coefficient", "is_derived", "abs_coef"])

    X = df[ALL_FEATURE_COLS].values
    y = df["engine_condition"].values
    X_scaled = StandardScaler().fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    coefs = model.coef_[0]
    result = pd.DataFrame({
        "feature":     ALL_FEATURE_COLS,
        "label":       [ALL_FEATURE_LABELS.get(c, c) for c in ALL_FEATURE_COLS],
        "coefficient": coefs,
        "is_derived":  [c in DERIVED_COLS for c in ALL_FEATURE_COLS],
    })
    result["abs_coef"] = result["coefficient"].abs()
    return result.sort_values("abs_coef", ascending=True).reset_index(drop=True)


def compute_health_score(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.Series:
    import numpy as np
    ref_oil  = np.sort(ref_df["oil_efficiency"].values)
    ref_cool = np.sort(ref_df["coolant_efficiency"].values)
    n = len(ref_oil)
    oil_pct  = pd.Series(
        np.searchsorted(ref_oil,  df["oil_efficiency"].values)  / n * 100,
        index=df.index,
    )
    cool_pct = pd.Series(
        np.searchsorted(ref_cool, df["coolant_efficiency"].values) / n * 100,
        index=df.index,
    )
    return (oil_pct + cool_pct) / 2
