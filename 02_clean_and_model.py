# ============================================================
# Myanmar Conflict Forecasting with ACLED Data
# Period: January 2021 - June 2025
#
# Models:
#   1. Negative Binomial regression
#   2. SARIMA time-series model
#
# Purpose:
#   Exploratory and benchmark conflict forecasting analysis.
#   Forecasts should not be interpreted as certain predictions.
# ============================================================


# ------------------------------------------------------------
# 1. Import libraries
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Create directory for generated figures
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. Load ACLED event-level data
# ------------------------------------------------------------

DATA_FILE = "myanmar_acled_2021_2025.csv"

df = pd.read_csv(DATA_FILE)

print("\n--- Original Data ---")
print(f"Object type: {type(df)}")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print("\nColumn names:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 3. Validate required variables
# ------------------------------------------------------------

required_columns = [
    "event_id_cnty",
    "event_date",
    "event_type",
    "admin1",
    "fatalities",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )


# ------------------------------------------------------------
# 4. Check duplicate event IDs
# ------------------------------------------------------------

duplicate_events = df["event_id_cnty"].duplicated().sum()

print(
    f"\nDuplicate event IDs detected: "
    f"{duplicate_events:,}"
)

if duplicate_events > 0:
    raise ValueError(
        "Duplicate ACLED event IDs were detected. "
        "Investigate before continuing."
    )


# ------------------------------------------------------------
# 5. Clean dates
# ------------------------------------------------------------

df["event_date"] = pd.to_datetime(
    df["event_date"],
    errors="coerce"
)

invalid_dates = df["event_date"].isna().sum()

if invalid_dates > 0:
    raise ValueError(
        f"{invalid_dates} invalid event dates detected."
    )

df = (
    df
    .sort_values("event_date")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 6. Convert numeric variables
# ------------------------------------------------------------

# Geographic variables:
# Missing coordinates should remain missing rather than
# being replaced with 0, because latitude/longitude = 0
# represents a real geographic location.

geographic_columns = [
    "latitude",
    "longitude",
    "geo_precision",
]

for column in geographic_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# Fatalities are converted separately.
# Missing fatalities are retained temporarily so that
# their presence can be inspected explicitly.

df["fatalities"] = pd.to_numeric(
    df["fatalities"],
    errors="coerce"
)

missing_fatalities = df["fatalities"].isna().sum()

print(
    f"Missing fatalities values: "
    f"{missing_fatalities:,}"
)

# For this aggregated exercise, missing fatality values
# are replaced with zero only after being reported.
# Reconsider this assumption if missing values indicate
# unknown rather than zero fatalities.

df["fatalities"] = df["fatalities"].fillna(0)


# ------------------------------------------------------------
# 7. Convert categorical variables
# ------------------------------------------------------------

category_columns = [
    "event_type",
    "admin1",
]

for column in category_columns:
    df[column] = df[column].astype("category")


# ------------------------------------------------------------
# 8. Create time variables
# ------------------------------------------------------------

df["year"] = df["event_date"].dt.year

df["year_month"] = (
    df["event_date"]
    .dt
    .to_period("M")
)


# ------------------------------------------------------------
# 9. Aggregate event data to ADMIN1-month level
# ------------------------------------------------------------

monthly_summary = (
    df
    .groupby(
        ["year_month", "admin1"],
        observed=True
    )
    .agg(
        total_events=(
            "event_id_cnty",
            "nunique"
        ),
        total_fatalities=(
            "fatalities",
            "sum"
        ),
        total_battles=(
            "event_type",
            lambda x: (
                x == "Battles"
            ).sum()
        ),
        total_explosions=(
            "event_type",
            lambda x: (
                x
                == "Explosions/Remote violence"
            ).sum()
        ),
        total_violence_civilians=(
            "event_type",
            lambda x: (
                x
                == "Violence against civilians"
            ).sum()
        ),
    )
    .reset_index()
)

# Convert monthly Period objects to timestamps.

monthly_summary["ds"] = (
    monthly_summary["year_month"]
    .dt
    .to_timestamp()
)

print("\n--- ADMIN1-Month Data ---")
print(monthly_summary.head())
print(
    f"ADMIN1-month observations: "
    f"{len(monthly_summary):,}"
)


# ------------------------------------------------------------
# 10. Aggregate ADMIN1 observations to national-month level
# ------------------------------------------------------------

national_monthly = (
    monthly_summary
    .groupby("ds", as_index=False)[
        [
            "total_events",
            "total_fatalities",
        ]
    ]
    .sum()
    .sort_values("ds")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 11. Verify monthly continuity
# ------------------------------------------------------------

expected_months = pd.date_range(
    start=national_monthly["ds"].min(),
    end=national_monthly["ds"].max(),
    freq="MS",
)

observed_months = pd.DatetimeIndex(
    national_monthly["ds"]
)

missing_months = expected_months.difference(
    observed_months
)

if len(missing_months) > 0:
    raise ValueError(
        "Missing calendar months detected:\n"
        f"{missing_months.tolist()}\n"
        "Do not automatically treat missing months "
        "as zero-event months."
    )


print("\n--- National Monthly Data ---")
print(national_monthly.head())
print(
    f"National monthly observations: "
    f"{len(national_monthly)}"
)


# ------------------------------------------------------------
# 12. Plot historical monthly conflict trend
# ------------------------------------------------------------

plt.figure(figsize=(11, 6))

plt.plot(
    national_monthly["ds"],
    national_monthly["total_events"],
    marker="o",
    linewidth=1.5
)

plt.title(
    "Monthly Political Violence Events in Myanmar "
    "(January 2021 - June 2025)"
)

plt.xlabel("Month")
plt.ylabel("Number of Events")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "monthly_conflict_trend.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 13. Descriptive statistics and raw overdispersion
# ------------------------------------------------------------

mean_events = (
    national_monthly["total_events"].mean()
)

var_events = (
    national_monthly["total_events"].var()
)

mean_fatalities = (
    national_monthly["total_fatalities"].mean()
)

var_fatalities = (
    national_monthly["total_fatalities"].var()
)

event_dispersion_ratio = (
    var_events / mean_events
)

fatality_dispersion_ratio = (
    var_fatalities / mean_fatalities
)


print("\n--- Descriptive Statistics ---")

print(
    f"Mean monthly events: "
    f"{mean_events:.2f}"
)

print(
    f"Variance in monthly events: "
    f"{var_events:.2f}"
)

print(
    f"Events variance/mean ratio: "
    f"{event_dispersion_ratio:.2f}"
)

print(
    f"Mean monthly fatalities: "
    f"{mean_fatalities:.2f}"
)

print(
    f"Variance in monthly fatalities: "
    f"{var_fatalities:.2f}"
)

print(
    f"Fatalities variance/mean ratio: "
    f"{fatality_dispersion_ratio:.2f}"
)


# ------------------------------------------------------------
# 14. Create one-month lag variables
# ------------------------------------------------------------

national_monthly["lag_events_1"] = (
    national_monthly["total_events"]
    .shift(1)
)

national_monthly["lag_fatalities_1"] = (
    national_monthly["total_fatalities"]
    .shift(1)
)


model_data = (
    national_monthly
    .dropna(
        subset=[
            "total_events",
            "lag_events_1",
            "lag_fatalities_1",
        ]
    )
    .copy()
)


# ------------------------------------------------------------
# 15. Fit Negative Binomial regression
# ------------------------------------------------------------

# Unlike the earlier GLM NegativeBinomial family call,
# this discrete Negative Binomial model estimates its
# dispersion parameter from the data.

negbi_formula = (
    "total_events "
    "~ lag_events_1 "
    "+ lag_fatalities_1"
)

negbi_model = (
    smf
    .negativebinomial(
        formula=negbi_formula,
        data=model_data,
    )
    .fit(
        disp=False,
        maxiter=1000,
    )
)


print(
    "\n--- Negative Binomial Regression ---"
)

print(
    negbi_model.summary()
)


# ------------------------------------------------------------
# 16. Define monthly national time series
# ------------------------------------------------------------

timeseries_data = (
    national_monthly
    .set_index("ds")["total_events"]
    .asfreq("MS")
)

if timeseries_data.isna().any():
    raise ValueError(
        "Missing observations remain after "
        "setting monthly frequency."
    )


# ------------------------------------------------------------
# 17. Six-month historical holdout
# ------------------------------------------------------------

FORECAST_HORIZON = 6

train_series = (
    timeseries_data
    .iloc[:-FORECAST_HORIZON]
)

test_series = (
    timeseries_data
    .iloc[-FORECAST_HORIZON:]
)


print("\n--- Backtest Split ---")

print(
    f"Training period: "
    f"{train_series.index.min().date()} "
    f"to "
    f"{train_series.index.max().date()}"
)

print(
    f"Test period: "
    f"{test_series.index.min().date()} "
    f"to "
    f"{test_series.index.max().date()}"
)


# ------------------------------------------------------------
# 18. Fit exploratory SARIMA benchmark on training data
# ------------------------------------------------------------

# This specification is a benchmark rather than proof that
# (1,1,1)x(1,1,1,12) is globally optimal.

sarima_train_model = SARIMAX(
    train_series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)

sarima_train_results = (
    sarima_train_model
    .fit(disp=False)
)


# ------------------------------------------------------------
# 19. Forecast the six held-out historical months
# ------------------------------------------------------------

backtest_forecast = (
    sarima_train_results
    .get_forecast(
        steps=FORECAST_HORIZON
    )
)

backtest_predictions = (
    backtest_forecast
    .predicted_mean
)

backtest_predictions.index = (
    test_series.index
)


# ------------------------------------------------------------
# 20. Calculate genuine out-of-sample errors
# ------------------------------------------------------------

backtest_errors = (
    test_series
    - backtest_predictions
)

sarima_mae = np.mean(
    np.abs(backtest_errors)
)

sarima_rmse = np.sqrt(
    np.mean(
        backtest_errors ** 2
    )
)


print(
    "\n--- SARIMA Six-Month Holdout Performance ---"
)

print(
    f"Out-of-sample MAE: "
    f"{sarima_mae:.2f}"
)

print(
    f"Out-of-sample RMSE: "
    f"{sarima_rmse:.2f}"
)


# ------------------------------------------------------------
# 21. Compare against a simple naive benchmark
# ------------------------------------------------------------

# Persistence forecast:
# every test month is predicted to equal the final
# observed event count in the training sample.

naive_predictions = pd.Series(
    train_series.iloc[-1],
    index=test_series.index,
    dtype=float,
)

naive_errors = (
    test_series
    - naive_predictions
)

naive_mae = np.mean(
    np.abs(naive_errors)
)

naive_rmse = np.sqrt(
    np.mean(
        naive_errors ** 2
    )
)


print(
    "\n--- Naive Benchmark Performance ---"
)

print(
    f"Naive MAE: "
    f"{naive_mae:.2f}"
)

print(
    f"Naive RMSE: "
    f"{naive_rmse:.2f}"
)


if sarima_rmse < naive_rmse:
    print(
        "\nSARIMA outperformed the naive "
        "benchmark on this holdout window."
    )
else:
    print(
        "\nSARIMA did NOT outperform the naive "
        "benchmark on this holdout window."
    )


# ------------------------------------------------------------
# 22. Create backtest comparison table
# ------------------------------------------------------------

backtest_df = pd.DataFrame(
    {
        "actual_events": test_series,
        "sarima_forecast": backtest_predictions,
        "naive_forecast": naive_predictions,
    }
)

backtest_df[
    "sarima_error"
] = (
    backtest_df["actual_events"]
    - backtest_df["sarima_forecast"]
)

backtest_df[
    "naive_error"
] = (
    backtest_df["actual_events"]
    - backtest_df["naive_forecast"]
)


print(
    "\n--- Backtest Forecast Comparison ---"
)

print(
    backtest_df.round(2)
)


# ------------------------------------------------------------
# 23. Fit SARIMA on the complete Jan 2021-Jun 2025 series
# ------------------------------------------------------------

final_sarima_model = SARIMAX(
    timeseries_data,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)

final_sarima_results = (
    final_sarima_model
    .fit(disp=False)
)


print(
    "\n--- Final SARIMA Model ---"
)

print(
    final_sarima_results.summary()
)


# ------------------------------------------------------------
# 24. Residual Ljung-Box test
# ------------------------------------------------------------

# Test whether residual autocorrelation remains.
# H0: residuals are independently distributed
# at the tested lags.

residuals = (
    final_sarima_results
    .resid
    .dropna()
)

ljung_box_results = (
    acorr_ljungbox(
        residuals,
        lags=[4, 8, 12],
        return_df=True,
    )
)


print(
    "\n--- Ljung-Box Residual Test ---"
)

print(
    ljung_box_results
)


# ------------------------------------------------------------
# 25. Historical fitted-value RMSE
# ------------------------------------------------------------

# This is reported only as an in-sample descriptive
# diagnostic. It is NOT treated as forecast accuracy.

historical_predictions = (
    final_sarima_results
    .predict(
        start=0,
        end=len(timeseries_data) - 1,
    )
)

historical_comparison = pd.concat(
    [
        timeseries_data.rename("actual"),
        historical_predictions.rename("predicted"),
    ],
    axis=1,
).dropna()

historical_errors = (
    historical_comparison["actual"]
    - historical_comparison["predicted"]
)

in_sample_rmse = np.sqrt(
    np.mean(
        historical_errors ** 2
    )
)


print(
    "\n--- In-Sample Diagnostic ---"
)

print(
    f"In-sample RMSE: "
    f"{in_sample_rmse:.2f}"
)


# ------------------------------------------------------------
# 26. Forecast July-December 2025
# ------------------------------------------------------------

future_forecast = (
    final_sarima_results
    .get_forecast(
        steps=FORECAST_HORIZON
    )
)

future_forecast_df = (
    future_forecast
    .conf_int()
    .copy()
)

future_forecast_df[
    "forecast"
] = (
    future_forecast
    .predicted_mean
)

future_forecast_df = (
    future_forecast_df[
        [
            "forecast",
            "lower total_events",
            "upper total_events",
        ]
    ]
)


print(
    "\n--- July-December 2025 "
    "SARIMA Forecast ---"
)

print(
    future_forecast_df.round(2)
)


# ------------------------------------------------------------
# 27. Plot residual diagnostics
# ------------------------------------------------------------

diagnostic_figure = final_sarima_results.plot_diagnostics(
    figsize=(10, 6)
)

plt.tight_layout()

diagnostic_figure.savefig(
    FIGURES_DIR / "sarima_diagnostics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 28. Plot historical data and future forecast
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    timeseries_data.index,
    timeseries_data.values,
    label="Observed events",
)

plt.plot(
    future_forecast_df.index,
    future_forecast_df["forecast"],
    label="SARIMA forecast",
)

plt.fill_between(
    future_forecast_df.index,
    future_forecast_df[
        "lower total_events"
    ],
    future_forecast_df[
        "upper total_events"
    ],
    alpha=0.2,
    label="95% confidence interval",
)

plt.title(
    "Myanmar Monthly Political Violence Events: "
    "SARIMA Forecast"
)

plt.xlabel("Month")
plt.ylabel("Number of events")
plt.legend()

plt.tight_layout()
plt.savefig(
    FIGURES_DIR / "sarima_forecast.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()