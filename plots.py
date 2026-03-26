import pandas as pd
import matplotlib.pyplot as plt


def plot_daily_cost_comparison(
    daily_greedy: pd.DataFrame,
    daily_milp: pd.DataFrame,
) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(
        pd.to_datetime(daily_greedy.index),
        daily_greedy["cost_pence"] / 100.0,
        label="Greedy (£/day)",
    )
    plt.plot(
        pd.to_datetime(daily_milp.index),
        daily_milp["cost_pence"] / 100.0,
        label="MILP (£/day)",
    )
    plt.title("Daily Electricity Cost (£): Greedy vs MILP")
    plt.xlabel("Date")
    plt.ylabel("Cost (£)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_soc_comparison(
    ts_greedy: pd.DataFrame,
    ts_milp: pd.DataFrame,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(
        ts_greedy.index,
        ts_greedy["soc_kwh"],
        label="Greedy SOC (kWh)",
        linestyle="--",
    )
    plt.plot(
        ts_milp.index,
        ts_milp["soc_kwh"],
        label="MILP SOC (kWh)",
    )
    plt.title("Battery State of Charge: Greedy vs MILP")
    plt.xlabel("Time")
    plt.ylabel("SOC (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_example_day_flows(
    ts_df: pd.DataFrame,
    date_str: str,
    title_prefix: str = "Scheduler",
) -> None:
    day = ts_df.loc[date_str:date_str]

    if len(day) == 0:
        raise ValueError(f"No data found for {date_str}")

    plt.figure(figsize=(12, 5))
    plt.plot(day.index, day["load_kwh"], label="Load (kWh)")
    plt.plot(day.index, day["solar_kwh"], label="Solar (kWh)")
    plt.plot(day.index, day["grid_import_kwh"], label="Grid Import (kWh)")
    plt.plot(day.index, day["grid_export_kwh"], label="Grid Export (kWh)")
    plt.plot(day.index, day["charge_kwh"], label="Battery Charge (kWh)")
    plt.plot(day.index, day["discharge_kwh"], label="Battery Discharge (kWh)")
    plt.title(f"{title_prefix} Energy Flows on {date_str}")
    plt.xlabel("Time")
    plt.ylabel("Energy per 30 min slot (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_example_day_prices(
    ts_df: pd.DataFrame,
    date_str: str,
    title_prefix: str = "Scheduler",
) -> None:
    day = ts_df.loc[date_str:date_str]

    if len(day) == 0:
        raise ValueError(f"No data found for {date_str}")

    plt.figure(figsize=(12, 4))
    plt.plot(day.index, day["import_price_p_per_kwh"], label="Import Price (p/kWh)")
    plt.plot(day.index, day["export_price_p_per_kwh"], label="Export Price (p/kWh)")
    plt.title(f"{title_prefix} Price Profile on {date_str}")
    plt.xlabel("Time")
    plt.ylabel("Price (p/kWh)")
    plt.legend()
    plt.tight_layout()
    plt.show()