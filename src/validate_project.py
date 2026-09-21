"""Check that the generated artefacts can support the demonstration application."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "monthly_water_consumption_cleaned.csv"
MODEL_PATH = ROOT / "models" / "best_monthly_consumption_model.joblib"
RESULTS_PATH = ROOT / "outputs" / "tables" / "model_comparison.csv"
FEATURES = [
    "district_area", "is_prv_area", "post_prv_installation", "private_tank_volume_litres",
    "pressure_before_kpa", "pressure_after_kpa", "pressure_change_kpa", "month_number",
    "month_index", "previous_month_consumption_m3",
]


def main() -> None:
    for path in [DATA_PATH, MODEL_PATH, RESULTS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required project output: {path}")

    data = pd.read_csv(DATA_PATH, low_memory=False)
    sample = data.dropna(subset=["previous_month_consumption_m3"])[FEATURES].head(1)
    if sample.empty:
        raise ValueError("No record with a previous-month value is available for a prediction check.")

    prediction = float(joblib.load(MODEL_PATH).predict(sample)[0])
    if not np.isfinite(prediction) or prediction < 0:
        raise ValueError("The saved model did not return a valid non-negative prediction.")

    results = pd.read_csv(RESULTS_PATH)
    required_columns = {"model", "test_mae", "test_rmse", "test_r2"}
    if results.empty or not required_columns.issubset(results.columns):
        raise ValueError("The model comparison table is incomplete.")

    print("Project validation passed.")
    print(f"Clean records checked: {len(data):,}")
    print(f"Saved model sample prediction: {prediction:.2f} m3")
    print(f"Best recorded model: {results.iloc[0]['model']}")


if __name__ == "__main__":
    main()
