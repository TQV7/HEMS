from __future__ import annotations

from pathlib import Path
import datetime as dt
import runpy

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Data loading
# --------------------------------------------------
def build_merged_df(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Build a merged half-hourly dataframe with:
    - load_kwh
    - import_price_p_per_kwh
    - export_price_p_per_kwh
    - solar_kwh
    """
    # Household load
    load_raw = pd.read_csv(data_dir / "house-load-2024.csv")
    load_raw["timestamp"] = pd.to_datetime(load_raw[" Start"], utc=True)

    load = (
        load_raw.rename(columns={"Consumption (kWh)": "load_kwh"})
        .set_index("timestamp")
        .sort_index()[["load_kwh"]]
    )

    # Agile prices
    prices_raw = pd.read_csv(data_dir / "agile-prices-2024.csv")
    prices_raw["timestamp"] = pd.to_datetime(
        prices_raw["Period from"], dayfirst=True
    ).dt.tz_localize("UTC")

    prices = (
        prices_raw.set_index("timestamp")
        .sort_index()
        .rename(
            columns={
                "Agile Import price (p/kWh)": "import_price_p_per_kwh",
                "Agile Export price (p/kWh)": "export_price_p_per_kwh",
            }
        )[["import_price_p_per_kwh", "export_price_p_per_kwh"]]
    )

    # Solar data from supervisor python file
    solar_namespace = runpy.run_path(str(data_dir / "solar_data.py"))
    actual_solar = solar_namespace["actualSolar"]

    rows = []
    for month in range(1, 13):
        for day in range(1, len(actual_solar[month])):
            for hour in range(24):
                ts = dt.datetime(2024, month, day, hour, tzinfo=dt.timezone.utc)
                hourly_kwh = actual_solar[month][day][hour]
                rows.append((ts, hourly_kwh))

    solar_hourly = (
        pd.DataFrame(rows, columns=["timestamp", "solar_hourly_kwh"])
        .set_index("timestamp")
        .sort_index()
    )

    # Convert hourly solar into half-hourly
    solar_halfhour = solar_hourly.resample("30min").ffill()
    solar_halfhour["solar_kwh"] = solar_halfhour["solar_hourly_kwh"] / 2.0
    solar_halfhour = solar_halfhour[["solar_kwh"]]

    merged = load.join(prices, how="inner").join(solar_halfhour, how="left").sort_index()
    merged["solar_kwh"] = merged["solar_kwh"].fillna(0.0)

    return merged


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def add_time_of_day_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_decimal"] = out.index.hour + out.index.minute / 60.0
    out["time_label"] = out.index.strftime("%H:%M")
    return out


def select_representative_solar_day(df: pd.DataFrame) -> str:
    """
    Pick the day with the highest total solar generation.
    Good for an explanatory 'typical solar day' figure.
    """
    daily_solar = df["solar_kwh"].groupby(df.index.strftime("%Y-%m-%d")).sum()
    return daily_solar.idxmax()


def plot_representative_day_profile(df: pd.DataFrame, day: str) -> None:
    """
    Plot load, solar, and import price on the same time axis.
    """
    day_df = df.loc[day:day].copy()
    day_df = add_time_of_day_columns(day_df)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(day_df["hour_decimal"], day_df["load_kwh"], label="Load (kWh)")
    ax1.plot(day_df["hour_decimal"], day_df["solar_kwh"], label="Solar (kWh)")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Energy per 30 min (kWh)")
    ax1.set_xticks(range(0, 25, 2))
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        day_df["hour_decimal"],
        day_df["import_price_p_per_kwh"],
        linestyle="--",
        label="Import price (p/kWh)",
    )
    ax2.set_ylabel("Price (p/kWh)")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.title(f"Representative Daily Household Profile ({day})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_representative_day_profile.png", dpi=300)
    plt.close()


def plot_average_daily_profile(df: pd.DataFrame) -> None:
    """
    Average half-hourly load, solar, and import price over the whole year.
    Very useful background figure.
    """
    tmp = add_time_of_day_columns(df)
    avg = tmp.groupby("hour_decimal")[
        ["load_kwh", "solar_kwh", "import_price_p_per_kwh"]
    ].mean()

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(avg.index, avg["load_kwh"], label="Average load (kWh)")
    ax1.plot(avg.index, avg["solar_kwh"], label="Average solar (kWh)")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Average energy per 30 min (kWh)")
    ax1.set_xticks(range(0, 25, 2))
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        avg.index,
        avg["import_price_p_per_kwh"],
        linestyle="--",
        label="Average import price (p/kWh)",
    )
    ax2.set_ylabel("Average price (p/kWh)")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.title("Average Daily Household Load, Solar, and Price Profile (2024)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_average_daily_profile.png", dpi=300)
    plt.close()


def plot_average_import_price_profile(df: pd.DataFrame) -> None:
    """
    Simple average import/export tariff profile across the day.
    Good for showing low overnight prices and daily variation.
    """
    tmp = add_time_of_day_columns(df)
    avg = tmp.groupby("hour_decimal")[
        ["import_price_p_per_kwh", "export_price_p_per_kwh"]
    ].mean()

    plt.figure(figsize=(12, 4))
    plt.plot(avg.index, avg["import_price_p_per_kwh"], label="Average import price")
    plt.plot(avg.index, avg["export_price_p_per_kwh"], label="Average export price")
    plt.xlabel("Hour of day")
    plt.ylabel("Price (p/kWh)")
    plt.xticks(range(0, 25, 2))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Average Daily Agile Tariff Profile (2024)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_average_tariff_profile.png", dpi=300)
    plt.close()


def plot_seasonal_average_profiles(df: pd.DataFrame) -> None:
    """
    Optional extra figure: average daily load/solar by season.
    Good if you want one more explanatory chart.
    """
    tmp = add_time_of_day_columns(df)
    month_to_season = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }
    tmp["season"] = tmp.index.month.map(month_to_season)

    for column, title, filename in [
        ("load_kwh", "Seasonal Average Household Load Profile", "figure_seasonal_load_profiles.png"),
        ("solar_kwh", "Seasonal Average Solar Generation Profile", "figure_seasonal_solar_profiles.png"),
    ]:
        plt.figure(figsize=(12, 5))
        seasonal_avg = tmp.groupby(["season", "hour_decimal"])[column].mean().unstack(0)

        # enforce sensible order
        seasonal_avg = seasonal_avg[["Winter", "Spring", "Summer", "Autumn"]]

        for season in seasonal_avg.columns:
            plt.plot(seasonal_avg.index, seasonal_avg[season], label=season)

        plt.xlabel("Hour of day")
        plt.ylabel("Average energy per 30 min (kWh)")
        plt.xticks(range(0, 25, 2))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, dpi=300)
        plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:
    merged = build_merged_df()

    representative_day = select_representative_solar_day(merged)
    print(f"Using representative solar day: {representative_day}")

    plot_representative_day_profile(merged, representative_day)
    plot_average_daily_profile(merged)
    plot_average_import_price_profile(merged)
    plot_seasonal_average_profiles(merged)

    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()