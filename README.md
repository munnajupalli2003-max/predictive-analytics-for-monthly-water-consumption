# Monthly Water Consumption Analytics

A reproducible MSc Big Data Analytics project that predicts monthly water consumption per connection using public data from Palmas, Tocantins, Brazil.

The repository includes a complete data pipeline, model comparison, a saved XGBoost model, the project outputs needed by the dashboard, and a Streamlit application for demonstrating the result.

## What the Project Does

- Converts eight district-area Excel sheets into one connection-month dataset.
- Cleans invalid or missing consumption records and records data-quality checks.
- Creates district, time, pressure, tank-volume and previous-month-consumption features.
- Compares Median Baseline, Linear Regression, Decision Tree, Random Forest and XGBoost models.
- Evaluates models with a chronological hold-out period, MAE, RMSE and R-squared.
- Loads the saved best model in a Streamlit dashboard without retraining it.

## Main Result

XGBoost achieved the lowest hold-out test RMSE in this project: **6.694 m3**. The result is specific to the supplied Palmas dataset and should be tested again before being used for another location.

## Repository Contents

```text
app.py                         Streamlit dashboard
src/run_analysis.py            Cleaning, analysis, modelling and output pipeline
src/validate_project.py        Checks project outputs and the saved model
data/raw/                      Public source workbook
data/processed/                Cleaned data used by the dashboard
models/                        Saved XGBoost model
outputs/tables/                Model results and data-quality summaries
outputs/figures/               Figures used in the dashboard
requirements.txt               Pinned Python dependencies
```

## Run the Dashboard on Windows

### First-time setup

Open PowerShell or Command Prompt and run:

```powershell
git clone https://github.com/munnajupalli2003-max/predictive-analytics-for-monthly-water-consumption.git
cd dissertation-project-code
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in a browser. Keep the terminal window open while using the dashboard. Press `Ctrl+C` in that terminal to stop the dashboard.

### Later use

After the first-time setup, run only:

```powershell
cd dissertation-project-code
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard uses the included cleaned data, saved model, tables and figures. It does **not** retrain the models when it starts.

## Run the Dashboard on macOS or Linux

```bash
git clone https://github.com/munnajupalli2003-max/predictive-analytics-for-monthly-water-consumption.git
cd dissertation-project-code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Rebuild the Analysis

The analysis is already complete. Rebuild it only if the raw workbook changes or if a new model run is required.

```powershell
.\.venv\Scripts\python.exe src\run_analysis.py
.\.venv\Scripts\python.exe src\validate_project.py
```

## Data Source and Attribution

The public source dataset is available from Figshare:

<https://figshare.com/articles/dataset/Water_consumption_data/16992127?file=31429828>

It is used under the Creative Commons Attribution 4.0 International licence:

<https://creativecommons.org/licenses/by/4.0/>

Please retain this attribution if reusing the included data.

## Limits

This is a research demonstration. It predicts monthly water consumption only for scenarios similar to the supplied Palmas study data. It is not a live utility system, does not detect leaks, and should not be used for another city without local data and new validation.
