from __future__ import annotations

from typing import Tuple
import pandas as pd

from models.scheduler_base import BatteryScheduler


def simulate_period(
    merged: pd.DataFrame,
    start_date: str,
    end_date: str,
    scheduler: BatteryScheduler,
    soc_init_kwh: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate battery operation day-by-day using the supplied scheduler.

    Args:
        merged:
            Half-hourly merged dataframe indexed by UTC timestamp.
        start_date:
            Inclusive start date, e.g. "2024-01-01".
        end_date:
            Inclusive end date, e.g. "2024-12-31".
        scheduler:
            Any BatteryScheduler subclass, e.g. GreedyScheduler or MilpScheduler.
        soc_init_kwh:
            Initial battery state of charge in kWh.

    Returns:
        timeseries_df:
            Concatenated per-half-hour results.
        daily_df:
            Per-day summary including cost and SOC progression.
    """
    if len(merged) == 0:
        raise ValueError("merged dataframe is empty")

    all_results = []
    daily_records = []

    soc = float(soc_init_kwh)

    # merged index is UTC
    days = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")

    for day in days:
        day_start = day
        day_end = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=30)

        day_df = merged.loc[day_start:day_end]

        if len(day_df) == 0:
            continue

        results, cost_pence, soc_end = scheduler.schedule_day(
            day_df=day_df,
            soc_init_kwh=soc,
        )

        results = results.copy()
        results["date"] = day.strftime("%Y-%m-%d")
        results["scheduler"] = scheduler.__class__.__name__
        all_results.append(results)

        daily_records.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "scheduler": scheduler.__class__.__name__,
                "cost_pence": float(cost_pence),
                "soc_start_kwh": float(soc),
                "soc_end_kwh": float(soc_end),
            }
        )

        soc = float(soc_end)  # carry SOC forward

    if not all_results:
        raise ValueError(f"No data found in range {start_date} to {end_date}")

    timeseries_df = pd.concat(all_results).sort_index()
    daily_df = pd.DataFrame(daily_records).set_index("date")

    return timeseries_df, daily_df