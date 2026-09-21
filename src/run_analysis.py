"""Analysis pipeline for the Figshare monthly water-consumption workbook.

The workbook is reshaped from one sheet per district metered area into a single
connection-month dataset before quality checks, visual analysis, and modelling.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

DATASET_URL = "https://figshare.com/articles/dataset/Water_consumption_data/16992127?file=31429828"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"
TARGET = "monthly_consumption_m3"


def find_workbook() -> Path:
    candidates = sorted(RAW_DIR.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(
            "No .xlsx workbook found in data/raw. Place water_consumption_raw_data.xlsx there."
        )
    preferred = [path for path in candidates if "water" in path.name.lower()]
    return preferred[0] if preferred else candidates[0]


def numeric_value(value: object) -> float:
    """Extract the first numeric value from an Excel cell, otherwise return NaN."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def parse_month_label(label: object) -> pd.Timestamp:
    text = str(label).strip().lower()
    replacements = {"ago": "aug", "set": "sep", "out": "oct", "dez": "dec", "fev": "feb"}
    for old, new in replacements.items():
        text = re.sub(rf"^{old}", new, text)
    parsed = pd.to_datetime(text, format="%b_%y", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
    return parsed


def sheet_metadata(sheet_name: str, sheet: pd.DataFrame) -> dict:
    is_prv_area = sheet_name.upper().startswith("PRV")
    # In the workbook, row 2 stores pressure values and row 3 stores the PRV date.
    pressure_before = numeric_value(sheet.iat[2, 2]) if sheet.shape[0] > 2 else np.nan
    pressure_after = numeric_value(sheet.iat[2, 3]) if sheet.shape[1] > 3 else np.nan
    installation_date = pd.to_datetime(sheet.iat[3, 2], errors="coerce") if sheet.shape[0] > 3 else pd.NaT
    pressure_label = str(sheet.iat[2, 1]) if sheet.shape[1] > 1 else ""
    unit_match = re.search(r"\(([^)]+)\)", pressure_label)
    pressure_unit = unit_match.group(1) if unit_match else np.nan
    pressure_multiplier = 9.80665 if str(pressure_unit).lower() == "mca" else 1.0
    return {
        "district_area": sheet_name.replace(" ", "").upper(),
        "is_prv_area": is_prv_area,
        "pressure_before_original": pressure_before,
        "pressure_after_original": pressure_after,
        "pressure_unit": pressure_unit,
        "pressure_before_kpa": pressure_before * pressure_multiplier,
        "pressure_after_kpa": pressure_after * pressure_multiplier,
        "pressure_change_kpa": (pressure_after - pressure_before) * pressure_multiplier if is_prv_area else np.nan,
        "prv_installation_date": installation_date,
    }


def reshape_workbook(workbook_path: Path) -> tuple[pd.DataFrame, dict]:
    excel = pd.ExcelFile(workbook_path)
    records: list[pd.DataFrame] = []
    sheet_log: dict[str, dict] = {}

    for sheet_name in excel.sheet_names:
        if sheet_name.strip().lower() == "info":
            continue
        raw_sheet = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None)
        if raw_sheet.shape[0] <= 7:
            continue

        metadata = sheet_metadata(sheet_name, raw_sheet)
        values = raw_sheet.iloc[6:].copy()
        values.columns = values.iloc[0].tolist()
        values = values.iloc[1:].copy()
        values = values.rename(columns={values.columns[0]: "connection_id", values.columns[1]: "private_tank_volume_raw"})

        month_columns = [column for column in values.columns[2:] if not pd.isna(parse_month_label(column))]
        if not month_columns:
            raise ValueError(f"No month columns were found in sheet '{sheet_name}'.")

        values = values[["connection_id", "private_tank_volume_raw", *month_columns]]
        values["connection_id"] = pd.to_numeric(values["connection_id"], errors="coerce")
        values = values.dropna(subset=["connection_id"])
        values["connection_id"] = values["connection_id"].astype(int)
        values["private_tank_volume_litres"] = values["private_tank_volume_raw"].map(numeric_value)
        long = values.melt(
            id_vars=["connection_id", "private_tank_volume_litres"],
            value_vars=month_columns,
            var_name="month_label",
            value_name=TARGET,
        )
        long["date"] = long["month_label"].map(parse_month_label)
        long[TARGET] = pd.to_numeric(long[TARGET], errors="coerce")
        for key, value in metadata.items():
            long[key] = value
        long["post_prv_installation"] = np.where(
            long["is_prv_area"] & (long["date"] >= long["prv_installation_date"]), 1, 0
        )
        long["connection_key"] = long["district_area"] + "_" + long["connection_id"].astype(str)
        long = long.drop(columns=["month_label"])
        records.append(long)
        sheet_log[sheet_name] = {
            "source_rows": int(len(values)),
            "month_columns": int(len(month_columns)),
            "long_rows": int(len(long)),
            **{key: (str(value.date()) if isinstance(value, pd.Timestamp) and not pd.isna(value) else value) for key, value in metadata.items()},
        }

    if not records:
        raise ValueError("No usable district-area sheets were found in the workbook.")
    combined = pd.concat(records, ignore_index=True)
    return combined, sheet_log


def quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[column].dtype) for column in df.columns],
            "missing_values": [int(df[column].isna().sum()) for column in df.columns],
            "missing_percent": [round(float(df[column].isna().mean() * 100), 2) for column in df.columns],
            "unique_values": [int(df[column].nunique(dropna=True)) for column in df.columns],
        }
    )


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cleaned = df.copy()
    rows_before = len(cleaned)
    duplicate_rows = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()
    invalid_target_rows = int((cleaned[TARGET].isna() | (cleaned[TARGET] < 0)).sum())
    cleaned = cleaned.dropna(subset=["date", TARGET])
    cleaned = cleaned[cleaned[TARGET] >= 0].copy()
    cleaned["year"] = cleaned["date"].dt.year
    cleaned["month_number"] = cleaned["date"].dt.month
    cleaned["month_name"] = cleaned["date"].dt.month_name()
    cleaned["month_index"] = (cleaned["date"].dt.year - cleaned["date"].dt.year.min()) * 12 + cleaned["date"].dt.month
    cleaned = cleaned.sort_values(["connection_key", "date"])
    cleaned["previous_month_consumption_m3"] = cleaned.groupby("connection_key")[TARGET].shift(1)
    cleaned["consumption_change_from_previous_m3"] = cleaned[TARGET] - cleaned["previous_month_consumption_m3"]

    q1, q3 = cleaned[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    possible_outliers = int(((cleaned[TARGET] < lower_fence) | (cleaned[TARGET] > upper_fence)).sum())

    log = {
        "rows_before_cleaning": int(rows_before),
        "duplicate_rows_removed": duplicate_rows,
        "invalid_or_missing_consumption_rows_removed": invalid_target_rows,
        "rows_after_cleaning": int(len(cleaned)),
        "columns_after_cleaning": int(cleaned.shape[1]),
        "possible_iqr_outliers_retained_for_review": possible_outliers,
        "outlier_rule": "No consumption rows were automatically removed; IQR flags are retained for transparent review.",
    }
    return cleaned, log


def save_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=TARGET, bins=40, kde=True, color="#2E74B5")
    plt.title("Distribution of Monthly Water Consumption per Connection")
    plt.xlabel("Monthly consumption (m3)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "monthly_consumption_distribution.png", dpi=180)
    plt.close()

    monthly = df.groupby("date", as_index=False)[TARGET].mean()
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=monthly, x="date", y=TARGET, marker="o", color="#2E74B5")
    plt.title("Average Monthly Water Consumption Over Time")
    plt.xlabel("Month")
    plt.ylabel("Average consumption per connection (m3)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "average_monthly_consumption_trend.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    order = df.groupby("district_area")[TARGET].median().sort_values().index
    sns.boxplot(data=df, x="district_area", y=TARGET, order=order, color="#4C956C")
    plt.title("Monthly Consumption by District Metered Area")
    plt.xlabel("District metered area")
    plt.ylabel("Monthly consumption per connection (m3)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "consumption_by_district_area.png", dpi=180)
    plt.close()

    labels = np.where(df["is_prv_area"], "Pressure-reduction area", "Comparison area")
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=labels, y=df[TARGET], color="#E9C46A")
    plt.title("Consumption in Pressure-Reduction and Comparison Areas")
    plt.xlabel("")
    plt.ylabel("Monthly consumption per connection (m3)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "consumption_by_prv_area_type.png", dpi=180)
    plt.close()

    numerical = [TARGET, "private_tank_volume_litres", "pressure_before_kpa", "pressure_after_kpa", "pressure_change_kpa", "month_index", "previous_month_consumption_m3"]
    numerical = [column for column in numerical if column in df.columns]
    correlation = df[numerical].corr(numeric_only=True)
    correlation.to_csv(TABLE_DIR / "correlation_matrix.csv")
    correlation[TARGET].drop(TARGET).sort_values(key=lambda series: series.abs(), ascending=False).to_frame("correlation_with_target").to_csv(TABLE_DIR / "target_correlations.csv")
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Correlation Matrix for Numerical Variables")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_matrix.png", dpi=180)
    plt.close()

    df.groupby(["district_area", "date"], as_index=False)[TARGET].mean().to_csv(
        TABLE_DIR / "monthly_consumption_by_district_area.csv", index=False
    )


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numerical = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical = features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numerical,
            ),
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                categorical,
            ),
        ],
        remainder="drop",
    )


def rmse(y_true: pd.Series, prediction: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, prediction)))


def evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, Pipeline]:
    model_data = df.dropna(subset=["previous_month_consumption_m3"]).copy()
    feature_columns = [
        "district_area",
        "is_prv_area",
        "post_prv_installation",
        "private_tank_volume_litres",
        "pressure_before_kpa",
        "pressure_after_kpa",
        "pressure_change_kpa",
        "month_number",
        "month_index",
        "previous_month_consumption_m3",
    ]
    features = model_data[feature_columns]
    target = model_data[TARGET]
    dates = model_data["date"]
    unique_dates = np.array(sorted(dates.unique()))
    cutoff = unique_dates[max(1, int(len(unique_dates) * 0.8)) - 1]
    train_mask = dates <= cutoff
    test_mask = dates > cutoff
    if not test_mask.any():
        raise ValueError("A later time period is needed for the hold-out test set.")

    x_train, x_test = features.loc[train_mask], features.loc[test_mask]
    y_train, y_test = target.loc[train_mask], target.loc[test_mask]
    preprocessor = build_preprocessor(features)
    models = {
        "Baseline Median": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=5, random_state=42, n_jobs=-1),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.9,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=1,
        )
    except ImportError:
        pass

    rows = []
    fitted: dict[str, Pipeline] = {}
    chronological = model_data.sort_values("date")
    cv_features = chronological[feature_columns]
    cv_target = chronological[TARGET]
    splitter = TimeSeriesSplit(n_splits=3)
    for name, model in models.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipeline.fit(x_train, y_train)
        train_prediction = pipeline.predict(x_train)
        test_prediction = pipeline.predict(x_test)
        cv_scores = []
        for train_index, validation_index in splitter.split(cv_features):
            cv_pipeline = Pipeline([("preprocess", build_preprocessor(cv_features)), ("model", model)])
            cv_pipeline.fit(cv_features.iloc[train_index], cv_target.iloc[train_index])
            cv_scores.append(rmse(cv_target.iloc[validation_index], cv_pipeline.predict(cv_features.iloc[validation_index])))
        rows.append(
            {
                "model": name,
                "train_mae": mean_absolute_error(y_train, train_prediction),
                "test_mae": mean_absolute_error(y_test, test_prediction),
                "train_rmse": rmse(y_train, train_prediction),
                "test_rmse": rmse(y_test, test_prediction),
                "train_r2": r2_score(y_train, train_prediction),
                "test_r2": r2_score(y_test, test_prediction),
                "time_series_cv_rmse_mean": float(np.mean(cv_scores)),
                "time_series_cv_rmse_std": float(np.std(cv_scores)),
                "training_rows": int(len(x_train)),
                "test_rows": int(len(x_test)),
                "test_period_start": str(pd.Timestamp(dates.loc[test_mask].min()).date()),
                "test_period_end": str(pd.Timestamp(dates.loc[test_mask].max()).date()),
            }
        )
        fitted[name] = pipeline

    results = pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)
    best_name = results.loc[0, "model"]
    best_pipeline = fitted[best_name]
    results.to_csv(TABLE_DIR / "model_comparison.csv", index=False)
    joblib.dump(best_pipeline, MODEL_DIR / "best_monthly_consumption_model.joblib")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=results, x="test_rmse", y="model", color="#2E74B5")
    plt.title("Model Comparison by Hold-Out Test RMSE")
    plt.xlabel("Test RMSE (m3)")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison_test_rmse.png", dpi=180)
    plt.close()
    return results, best_pipeline


def write_report(cleaning_log: dict, results: pd.DataFrame) -> None:
    best = results.iloc[0]
    lines = [
        "# Technical Analysis Report: Monthly Water Consumption",
        "",
        "## Dataset",
        "",
        "- Source: Figshare Water consumption data dataset.",
        f"- Dataset link: {DATASET_URL}",
        f"- Licence: CC BY 4.0 ({LICENCE_URL})",
        "- Unit of analysis: monthly water consumption per meter connection.",
        "",
        "## Cleaning Summary",
        "",
    ]
    lines.extend(f"- {key.replace('_', ' ').title()}: {value}" for key, value in cleaning_log.items())
    lines.extend(
        [
            "",
            "## Evaluation Design",
            "",
            "- The models use a chronological training/test split: earlier months for training and later months for testing.",
            "- A three-split time-series cross-validation check is included to assess stability.",
            "- The previous month's consumption is used as a predictor; the first observed month for each connection is excluded from modelling because it has no prior-month value.",
            "",
            "## Best Model",
            "",
            f"The best hold-out test model was {best['model']}.",
            f"- Test MAE: {best['test_mae']:.3f} m3",
            f"- Test RMSE: {best['test_rmse']:.3f} m3",
            f"- Test R-squared: {best['test_r2']:.3f}",
            f"- Mean time-series cross-validation RMSE: {best['time_series_cv_rmse_mean']:.3f} m3",
            "",
            "## Model Comparison",
            "",
            results.round(4).to_csv(index=False),
            "",
            "## Interpretation Caution",
            "",
            "These results describe the supplied dataset. They should not be presented as causal proof that pressure or tank volume alone changes consumption. The dataset is secondary data, and unmeasured factors may affect the results.",
        ]
    )
    (REPORT_DIR / "Technical_Analysis_Report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for folder in [PROCESSED_DIR, FIGURE_DIR, TABLE_DIR, REPORT_DIR, MODEL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    workbook_path = find_workbook()
    raw, sheet_log = reshape_workbook(workbook_path)
    quality_summary(raw).to_csv(TABLE_DIR / "data_quality_before_cleaning.csv", index=False)
    cleaned, cleaning_log = clean_data(raw)
    cleaned.to_csv(PROCESSED_DIR / "monthly_water_consumption_cleaned.csv", index=False)
    quality_summary(cleaned).to_csv(TABLE_DIR / "data_quality_after_cleaning.csv", index=False)
    (TABLE_DIR / "source_sheet_summary.json").write_text(json.dumps(sheet_log, indent=2, default=str), encoding="utf-8")
    (TABLE_DIR / "cleaning_log.json").write_text(json.dumps(cleaning_log, indent=2), encoding="utf-8")
    save_eda(cleaned)
    results, _ = evaluate_models(cleaned)
    write_report(cleaning_log, results)
    print("Analysis complete.")
    print(f"Cleaned dataset: {PROCESSED_DIR / 'monthly_water_consumption_cleaned.csv'}")
    print(f"Model comparison: {TABLE_DIR / 'model_comparison.csv'}")
    print(f"Technical report: {REPORT_DIR / 'Technical_Analysis_Report.md'}")


if __name__ == "__main__":
    main()
