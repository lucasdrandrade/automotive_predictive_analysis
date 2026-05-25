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

CONDITION_LABELS = {1: "Healthy", 0: "Faulty"}
CONDITION_COLORS = {1: "#27ae60", 0: "#e74c3c"}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_MAP)
    df = df.reset_index(drop=True)
    return df


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
