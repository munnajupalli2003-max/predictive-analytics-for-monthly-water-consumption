"""Streamlit demonstration for the monthly water-consumption analysis project."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / \
    "monthly_water_consumption_cleaned.csv"
MODEL_PATH = ROOT / "models" / "best_monthly_consumption_model.joblib"
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"

DATASET_URL = "https://figshare.com/articles/dataset/Water_consumption_data/16992127?file=31429828"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"
PAPER_URL = "https://ascelibrary.org/doi/abs/10.1061/%28ASCE%29WR.1943-5452.0001627"
TARGET = "monthly_consumption_m3"

st.set_page_config(page_title="Water Consumption Analytics",
                   page_icon="W", layout="wide")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, parse_dates=[
                       "date", "prv_installation_date"])
    return data


@st.cache_data(show_spinner=False)
def load_model_results() -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / "model_comparison.csv")


@st.cache_data(show_spinner=False)
def load_area_metadata() -> dict[str, dict]:
    with (TABLE_DIR / "source_sheet_summary.json").open(encoding="utf-8") as file:
        raw_metadata = json.load(file)
    return {details["district_area"]: details for details in raw_metadata.values()}


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


def project_ready() -> bool:
    return all(
        path.exists()
        for path in [DATA_PATH, MODEL_PATH, TABLE_DIR / "model_comparison.csv", TABLE_DIR / "source_sheet_summary.json"]
    )


def format_number(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def filter_data(data: pd.DataFrame, areas: list[str], start: date, end: date) -> pd.DataFrame:
    return data[
        data["district_area"].isin(areas)
        & data["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    ].copy()


def historical_message(area_data: pd.DataFrame, prediction: float) -> str:
    lower, upper = area_data[TARGET].quantile([0.25, 0.75])
    if prediction < lower:
        return "This is below the usual range for this area in the study data."
    if prediction > upper:
        return "This is above the usual range for this area in the study data."
    return "This is within the middle 50% of values seen in this area in the study data."


def show_figure(filename: str, caption: str) -> None:
    """Show a saved project figure when it is available."""
    figure_path = FIGURE_DIR / filename
    if figure_path.exists():
        st.image(str(figure_path), caption=caption, use_container_width=True)


st.markdown(
    """
    <style>
    .block-container {max-width: 1260px; padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {border-left: 3px solid #0f766e; padding-left: 0.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Monthly Water Consumption Analytics")
st.caption(
    "Project showcase using the public Palmas, Brazil water-consumption dataset.")

if not project_ready():
    st.error("The analysis files are not ready yet. Run `python src/run_analysis.py` from the project folder, then refresh this page.")
    st.stop()

data = load_data()
results = load_model_results()
metadata = load_area_metadata()

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a page",
        ["Project showcase", "Data evidence", "Data preparation",
            "Model performance", "Live prediction", "Evaluation notes"],
    )
    st.divider()
    st.caption("Public sources")
    st.link_button("Dataset source", DATASET_URL, use_container_width=True)
    st.link_button("CC BY 4.0 licence", LICENCE_URL, use_container_width=True)
    st.link_button("Related study", PAPER_URL, use_container_width=True)

if page == "Project showcase":
    st.subheader("The problem we worked on")
    st.write(
        "Water providers need a clear view of how much water is likely to be used next month. "
        "This project prepares real water-consumption data, compares machine-learning models, and gives a simple estimate for a connection in the study area."
    )
    first_date, last_date = data["date"].min(
    ).date(), data["date"].max().date()
    first, second, third, fourth = st.columns(4)
    first.metric("Clean records", f"{len(data):,}")
    second.metric("District areas", str(data["district_area"].nunique()))
    third.metric("Study period", f"{first_date:%Y}-{last_date:%Y}")
    best = results.iloc[0]
    fourth.metric("Best test RMSE", f"{best['test_rmse']:.2f} m3")

    st.subheader("What we completed")
    left, right, third, fourth = st.columns(4)
    with left:
        st.markdown(
            "**1. Prepared data**\n\nChanged eight Excel sheets into one clean dataset.")
    with right:
        st.markdown(
            "**2. Checked quality**\n\nRemoved invalid records and recorded possible outliers for review.")
    with third:
        st.markdown(
            "**3. Built models**\n\nCompared a baseline with four machine-learning models.")
    with fourth:
        st.markdown(
            "**4. Built a demo**\n\nCreated this tool to show results and make study-context estimates.")

    st.subheader("Main results")
    result_left, result_right = st.columns(2)
    with result_left:
        show_figure("average_monthly_consumption_trend.png",
                    "Average monthly water consumption in the supplied dataset")
    with result_right:
        show_figure("model_comparison_test_rmse.png",
                    "Comparison of model error on later unseen months")

    st.subheader("Key findings")
    findings = pd.DataFrame(
        [
            {"Finding": "Best model",
                "Evidence": f"{best['model']} had the lowest test RMSE: {best['test_rmse']:.2f} m3."},
            {"Finding": "Strongest pattern",
                "Evidence": "Previous month's consumption had the strongest numerical relationship with current consumption (correlation 0.767)."},
            {"Finding": "Different areas",
                "Evidence": "The eight district areas have different consumption distributions, so district area is included in the model."},
            {"Finding": "Useful output",
                "Evidence": "The live page gives an estimate and compares it with the historical range in the selected area."},
        ]
    )
    st.dataframe(findings, use_container_width=True, hide_index=True)

    st.info(
        "Models were trained on eight district metered areas in Palmas, Tocantins, Brazil. "
        "Predictions are most meaningful when the inputs are similar to this study context."
    )

elif page == "Data evidence":
    st.subheader("Evidence from the cleaned data")
    st.write("Use these charts and filters to show what was found before modelling.")
    areas = sorted(data["district_area"].unique())
    selection, period = st.columns([1, 2])
    with selection:
        chosen_areas = st.multiselect("District areas", areas, default=areas)
    with period:
        chosen_period = st.date_input(
            "Date range",
            value=(data["date"].min().date(), data["date"].max().date()),
            min_value=data["date"].min().date(),
            max_value=data["date"].max().date(),
        )
    if not chosen_areas or not isinstance(chosen_period, tuple) or len(chosen_period) != 2:
        st.warning("Choose at least one area and a start and end date.")
        st.stop()
    filtered = filter_data(
        data, chosen_areas, chosen_period[0], chosen_period[1])
    if filtered.empty:
        st.warning("There are no records for this selection.")
        st.stop()

    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Average use", f"{filtered[TARGET].mean():.2f} m3")
    metric_two.metric("Median use", f"{filtered[TARGET].median():.2f} m3")
    metric_three.metric(
        "Connections", f"{filtered['connection_key'].nunique():,}")

    monthly = filtered.groupby(["date", "district_area"], as_index=False)[
        TARGET].mean()
    st.plotly_chart(
        px.line(
            monthly, x="date", y=TARGET, color="district_area", markers=True,
            labels={"date": "Month",
                    TARGET: "Average monthly use (m3)", "district_area": "Area"},
            title="Average monthly consumption over time",
        ),
        use_container_width=True,
    )
    area_order = filtered.groupby("district_area")[
        TARGET].median().sort_values().index
    st.plotly_chart(
        px.box(
            filtered, x="district_area", y=TARGET, color="district_area", category_orders={"district_area": list(area_order)},
            labels={"district_area": "Area", TARGET: "Monthly use (m3)"}, title="Consumption distribution by area",
        ),
        use_container_width=True,
    )
    visual_left, visual_right = st.columns(2)
    with visual_left:
        show_figure("monthly_consumption_distribution.png",
                    "Distribution of monthly consumption per connection")
    with visual_right:
        show_figure("consumption_by_district_area.png",
                    "Consumption differences across the eight district areas")
    with st.expander("View selected records"):
        st.dataframe(filtered[["district_area", "connection_id", "date", TARGET, "private_tank_volume_litres"]].sort_values(
            "date"), use_container_width=True, hide_index=True)

elif page == "Data preparation":
    st.subheader("Data preparation")
    st.write("The original workbook has one sheet for each area. The pipeline reshapes it into one connection-month dataset before analysis.")
    stages = st.columns(4)
    stages[0].markdown(
        "**1. Read**\n\nRead the Excel sheets and their month columns.")
    stages[1].markdown(
        "**2. Reshape**\n\nCreate one row for every connection and month.")
    stages[2].markdown(
        "**3. Check**\n\nRemove duplicates and invalid or missing consumption values.")
    stages[3].markdown(
        "**4. Prepare**\n\nCreate time and previous-month-use features for modelling.")

    with (TABLE_DIR / "cleaning_log.json").open(encoding="utf-8") as file:
        cleaning_log = json.load(file)
    cleaning = pd.DataFrame(
        [{"Check": key.replace("_", " ").capitalize(), "Result": value}
         for key, value in cleaning_log.items()]
    )
    st.subheader("Cleaning record")
    st.dataframe(cleaning, use_container_width=True, hide_index=True)
    st.caption(
        "Potential outliers are retained and counted for review; they are not automatically deleted.")

    summary = pd.DataFrame(metadata.values())
    display_columns = ["district_area", "is_prv_area", "source_rows",
                       "month_columns", "pressure_unit", "prv_installation_date"]
    st.subheader("Source-sheet summary")
    st.dataframe(summary[display_columns].rename(columns={
                 "is_prv_area": "pressure-reduction area"}), use_container_width=True, hide_index=True)
    st.subheader("Feature relationships")
    relationship_left, relationship_right = st.columns(2)
    with relationship_left:
        show_figure("correlation_matrix.png",
                    "Correlation between numerical variables in the cleaned data")
    with relationship_right:
        correlations = pd.read_csv(TABLE_DIR / "target_correlations.csv")
        correlations = correlations.rename(columns={
                                           "Unnamed: 0": "Variable", "correlation_with_target": "Correlation with monthly consumption"})
        correlations["Correlation with monthly consumption"] = correlations["Correlation with monthly consumption"].round(
            3)
        st.dataframe(correlations, use_container_width=True, hide_index=True)
        st.caption(
            "A correlation shows a pattern in this dataset. It does not prove that one feature causes the other.")

elif page == "Model performance":
    st.subheader("Model performance")
    st.write("Each model is trained using earlier months and tested using later months. This better reflects a real forecasting situation.")
    best = results.iloc[0]
    one, two, three = st.columns(3)
    one.metric("Best model", best["model"])
    two.metric("Test MAE", f"{best['test_mae']:.2f} m3")
    three.metric("Test R-squared", f"{best['test_r2']:.3f}")

    chart_data = results.sort_values("test_rmse")
    st.plotly_chart(
        px.bar(chart_data, x="test_rmse", y="model", orientation="h", text_auto=".2f", color="test_rmse", color_continuous_scale="Teal",
               labels={"test_rmse": "Hold-out test RMSE (m3)", "model": "Model"}, title="Lower error is better"),
        use_container_width=True,
    )
    show_figure("model_comparison_test_rmse.png",
                "Saved model-comparison figure produced by the analysis pipeline")
    table = results[["model", "test_mae", "test_rmse", "test_r2",
                     "time_series_cv_rmse_mean", "test_period_start", "test_period_end"]].copy()
    st.dataframe(table.round(3), use_container_width=True, hide_index=True)
    st.caption("RMSE gives extra weight to larger errors. R-squared shows how much variation in the held-out data is explained by the model.")
    st.success(f"This dashboard automatically uses the  trained {best['model']} model. It does not train the model again when the dashboard is opened.")

elif page == "Live prediction":
    st.subheader("Study-context prediction")
    st.write("Enter a scenario similar to the supplied study data. The result is an estimate, not a decision on its own.")
    areas = sorted(metadata)
    default_area = areas[0]
    area = st.selectbox("District metered area", areas,
                        index=areas.index(default_area))
    area_data = data[data["district_area"] == area]
    area_meta = metadata[area]
    default_date = data["date"].max().date()
    col_one, col_two = st.columns(2)
    with col_one:
        prediction_date = st.date_input(
            "Month to estimate", value=default_date)
        previous_use = st.number_input(
            "Previous month's consumption (m3)", min_value=0.0,
            value=float(area_data["previous_month_consumption_m3"].median()), step=1.0,
        )
    with col_two:
        tank_volume = st.number_input(
            "Private tank volume (litres)", min_value=0.0,
            value=float(area_data["private_tank_volume_litres"].median()), step=100.0,
        )
        st.text_input(
            "Area type", "Pressure-reduction area" if area_meta["is_prv_area"] else "Comparison area", disabled=True)

    if st.button("Estimate monthly consumption", type="primary"):
        reference_date = data["date"].min()
        selected_date = pd.Timestamp(prediction_date)
        installation = pd.to_datetime(
            area_meta["prv_installation_date"], errors="coerce")
        post_installation = int(bool(area_meta["is_prv_area"]) and pd.notna(
            installation) and selected_date >= installation)
        model_input = pd.DataFrame([{
            "district_area": area,
            "is_prv_area": int(bool(area_meta["is_prv_area"])),
            "post_prv_installation": post_installation,
            "private_tank_volume_litres": tank_volume,
            "pressure_before_kpa": area_meta["pressure_before_kpa"],
            "pressure_after_kpa": area_meta["pressure_after_kpa"],
            "pressure_change_kpa": area_meta["pressure_change_kpa"],
            "month_number": selected_date.month,
            "month_index": (selected_date.year - reference_date.year) * 12 + selected_date.month,
            "previous_month_consumption_m3": previous_use,
        }])
        prediction = max(0.0, float(load_model().predict(model_input)[0]))
        difference = prediction - previous_use
        first, second = st.columns(2)
        first.metric("Estimated monthly consumption",
                     f"{prediction:.2f} m3", delta=f"{difference:+.2f} m3 compared with previous month")
        second.metric("Historical area median",
                      f"{area_data[TARGET].median():.2f} m3")
        st.info(historical_message(area_data, prediction))
      #  st.caption("For use outside the study area or time period, collect local data and retrain or validate the model before relying on the estimate.")

else:
    st.subheader("Evaluation and responsible use")
    st.write("The project is intended as decision support. It should help people ask better questions, not replace local engineering judgement or customer knowledge.")
    st.markdown("**Suggested evaluation activities**")
    st.markdown(
        "- Ask water-sector professionals whether the pages are clear and relevant to their work.")
    st.markdown(
        "- Ask whether the prediction inputs are realistic and understandable.")
    st.markdown(
        "- Record suggestions about extra data that would make the tool more useful locally.")
    st.markdown(
        "- Use the approved university evaluation process before collecting any responses.")
    st.warning("This tool should not be used to take decisions without professional review.")
    st.subheader("Known limitations")
    st.markdown(
        "- The dataset is secondary data from eight areas in Palmas, Brazil.")
    st.markdown(
        "- The model cannot prove that a feature causes higher or lower water use.")
    st.markdown(
        "- Missing pressure information in comparison areas is handled by the model's preprocessing step.")
    st.markdown(
        "- A new area needs local data and validation before the model is used there.")
