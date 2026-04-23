from pathlib import Path
import pandas as pd

from config import CONFIG, BatteryConfig
from data import build_merged_df
from simulate import simulate_period
from models.greedy import GreedyScheduler
from models.milp import MilpScheduler
from models.milpExportAllowed import MilpExportAllowedScheduler
from plots import (
    plot_daily_cost_comparison,
    plot_soc_comparison,
    plot_example_day_flows,
    plot_example_day_prices,
    plot_price_vs_battery,
)


def build_monthly_breakdown(daily_greedy, daily_milp, daily_milp_export):
    monthly_greedy = daily_greedy.copy()
    monthly_milp = daily_milp.copy()
    monthly_milp_export = daily_milp_export.copy()

    monthly_greedy.index = pd.to_datetime(monthly_greedy.index)
    monthly_milp.index = pd.to_datetime(monthly_milp.index)
    monthly_milp_export.index = pd.to_datetime(monthly_milp_export.index)

    monthly_g = monthly_greedy["cost_pence"].resample("ME").sum() / 100.0
    monthly_m = monthly_milp["cost_pence"].resample("ME").sum() / 100.0
    monthly_me = monthly_milp_export["cost_pence"].resample("ME").sum() / 100.0

    out = pd.DataFrame({
        "greedy_cost_gbp": monthly_g,
        "milp_no_export_cost_gbp": monthly_m,
        "milp_export_allowed_cost_gbp": monthly_me,
    })

    out["savings_no_export_gbp"] = out["greedy_cost_gbp"] - out["milp_no_export_cost_gbp"]
    out["savings_no_export_pct"] = (
        out["savings_no_export_gbp"] / out["greedy_cost_gbp"] * 100.0
    )

    out["savings_export_allowed_gbp"] = (
        out["greedy_cost_gbp"] - out["milp_export_allowed_cost_gbp"]
    )
    out["savings_export_allowed_pct"] = (
        out["savings_export_allowed_gbp"] / out["greedy_cost_gbp"] * 100.0
    )

    out.index = out.index.strftime("%Y-%m")
    return out


def run_battery_sweep(merged, start_date, end_date, capacities_kwh):
    results = []

    for cap in capacities_kwh:
        print(f"\n===== Running battery size: {cap} kWh =====")

        battery = BatteryConfig(
            capacity_kwh=cap,
            max_charge_kw=CONFIG.battery.max_charge_kw,
            max_discharge_kw=CONFIG.battery.max_discharge_kw,
            charge_efficiency=CONFIG.battery.charge_efficiency,
            discharge_efficiency=CONFIG.battery.discharge_efficiency,
            min_soc_kwh=CONFIG.battery.min_soc_kwh,
            initial_soc_kwh=cap * 0.5,
            timestep_hours=CONFIG.battery.timestep_hours,
        )

        greedy_scheduler = GreedyScheduler(battery)
        milp_scheduler = MilpScheduler(battery)

        _, daily_greedy = simulate_period(
            merged=merged,
            start_date=start_date,
            end_date=end_date,
            scheduler=greedy_scheduler,
            soc_init_kwh=battery.initial_soc_kwh,
        )

        _, daily_milp = simulate_period(
            merged=merged,
            start_date=start_date,
            end_date=end_date,
            scheduler=milp_scheduler,
            soc_init_kwh=battery.initial_soc_kwh,
        )

        greedy_total_pence = daily_greedy["cost_pence"].sum()
        milp_total_pence = daily_milp["cost_pence"].sum()

        greedy_total_gbp = greedy_total_pence / 100.0
        milp_total_gbp = milp_total_pence / 100.0
        savings_gbp = greedy_total_gbp - milp_total_gbp
        savings_pct = (
            (greedy_total_pence - milp_total_pence) / greedy_total_pence * 100.0
            if greedy_total_pence != 0
            else 0.0
        )

        results.append(
            {
                "battery_kwh": cap,
                "greedy_cost_gbp": greedy_total_gbp,
                "milp_cost_gbp": milp_total_gbp,
                "savings_gbp": savings_gbp,
                "savings_pct": savings_pct,
            }
        )

    return pd.DataFrame(results)


def main():
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    example_day = "2024-03-02"


    outputs_dir = Path("outputs")
    results_dir = outputs_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)


    merged = build_merged_df()
    print("Merged dataset:", merged.shape)
    print(merged.head())


    greedy_scheduler = GreedyScheduler(CONFIG.battery)
    milp_scheduler = MilpScheduler(CONFIG.battery)
    milp_export_allowed_scheduler = MilpExportAllowedScheduler(CONFIG.battery)


    ts_greedy, daily_greedy = simulate_period(
        merged=merged,
        start_date=start_date,
        end_date=end_date,
        scheduler=greedy_scheduler,
        soc_init_kwh=CONFIG.battery.initial_soc_kwh,
    )

    ts_milp, daily_milp = simulate_period(
        merged=merged,
        start_date=start_date,
        end_date=end_date,
        scheduler=milp_scheduler,
        soc_init_kwh=CONFIG.battery.initial_soc_kwh,
    )

    ts_milp_export, daily_milp_export = simulate_period(
        merged=merged,
        start_date=start_date,
        end_date=end_date,
        scheduler=milp_export_allowed_scheduler,
        soc_init_kwh=CONFIG.battery.initial_soc_kwh,
    )


    daily_greedy.to_csv(results_dir / "daily_greedy.csv")
    daily_milp.to_csv(results_dir / "daily_milp_no_export.csv")
    daily_milp_export.to_csv(results_dir / "daily_milp_export_allowed.csv")

    ts_greedy.to_csv(results_dir / "timeseries_greedy.csv")
    ts_milp.to_csv(results_dir / "timeseries_milp_no_export.csv")
    ts_milp_export.to_csv(results_dir / "timeseries_milp_export_allowed.csv")

    print("\nSaved baseline results to outputs/results/")


    monthly_breakdown = build_monthly_breakdown(
        daily_greedy=daily_greedy,
        daily_milp=daily_milp,
        daily_milp_export=daily_milp_export,
    )

    monthly_breakdown.to_csv(results_dir / "monthly_breakdown.csv")

    print("\n=== MONTHLY BREAKDOWN ===")
    print(monthly_breakdown)


    greedy_total_pence = daily_greedy["cost_pence"].sum()
    milp_total_pence = daily_milp["cost_pence"].sum()
    milp_export_total_pence = daily_milp_export["cost_pence"].sum()

    greedy_total_gbp = greedy_total_pence / 100.0
    milp_total_gbp = milp_total_pence / 100.0
    milp_export_total_gbp = milp_export_total_pence / 100.0

    savings_gbp = greedy_total_gbp - milp_total_gbp
    savings_pct = (
        (greedy_total_pence - milp_total_pence) / greedy_total_pence * 100.0
        if greedy_total_pence != 0
        else 0.0
    )

    savings_export_gbp = greedy_total_gbp - milp_export_total_gbp
    savings_export_pct = (
        (greedy_total_pence - milp_export_total_pence) / greedy_total_pence * 100.0
        if greedy_total_pence != 0
        else 0.0
    )

    print("\n=== DAILY SUMMARY (GREEDY) ===")
    print(daily_greedy)
    print(f"Total cost (Greedy): £{greedy_total_gbp:.2f}")

    print("\n=== DAILY SUMMARY (MILP - NO BATTERY EXPORT) ===")
    print(daily_milp)
    print(f"Total cost (MILP no export): £{milp_total_gbp:.2f}")

    print("\n=== DAILY SUMMARY (MILP - EXPORT ALLOWED) ===")
    print(daily_milp_export)
    print(f"Total cost (MILP export allowed): £{milp_export_total_gbp:.2f}")

    print("\n=== SAVINGS VS GREEDY ===")
    print(f"No-export MILP savings: £{savings_gbp:.2f} ({savings_pct:.2f}%)")
    print(f"Export-allowed MILP savings: £{savings_export_gbp:.2f} ({savings_export_pct:.2f}%)")


    capacities = [5, 10, 15]
    sweep_df = run_battery_sweep(
        merged=merged,
        start_date=start_date,
        end_date=end_date,
        capacities_kwh=capacities,
    )

    sweep_df.to_csv(results_dir / "battery_sweep_no_export.csv", index=False)

    print("\n=== BATTERY SWEEP (NO-EXPORT MILP) ===")
    print(sweep_df)


    plot_daily_cost_comparison(daily_greedy, daily_milp)
    plot_soc_comparison(ts_greedy, ts_milp)
    plot_price_vs_battery(ts_milp, "2024-01-15")

    plot_example_day_flows(ts_greedy, example_day, title_prefix="Greedy")
    plot_example_day_flows(ts_milp, example_day, title_prefix="MILP No Export")
    plot_example_day_prices(ts_milp, example_day, title_prefix="MILP No Export")


if __name__ == "__main__":
    main()