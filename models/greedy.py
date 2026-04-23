from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG, BatteryConfig
from .scheduler_base import BatteryScheduler


class GreedyScheduler(BatteryScheduler):
    """
    Greedy baseline controller (no forecasting).

    Priority:
      1) Use solar to meet load
      2) Charge battery with excess solar (up to limits), export leftover
      3) Discharge battery to cover remaining load (up to limits), import leftover

    No grid-charging.
    """

    def __init__(self, battery_config: BatteryConfig | None = None):
        super().__init__(battery_config=battery_config)

    def schedule_day(
        self,
        day_df: pd.DataFrame,
        soc_init_kwh: float,
    ) -> tuple[pd.DataFrame, float, float]:
        required = {
            CONFIG.data.load_column,
            CONFIG.data.solar_column,
            CONFIG.tariff.import_price_column,
            CONFIG.tariff.export_price_column,
        }
        missing = required - set(day_df.columns)
        if missing:
            raise ValueError(f"day_df missing columns: {sorted(missing)}")
        if len(day_df) == 0:
            raise ValueError("day_df is empty")

        df = day_df.copy()
        df[CONFIG.data.solar_column] = df[CONFIG.data.solar_column].fillna(0.0)

        T = len(df)
        e_max_charge = self.battery.max_charge_kwh_per_slot
        e_max_discharge = self.battery.max_discharge_kwh_per_slot

        load = df[CONFIG.data.load_column].to_numpy(dtype=float)
        solar = df[CONFIG.data.solar_column].to_numpy(dtype=float)
        p_imp = df[CONFIG.tariff.import_price_column].to_numpy(dtype=float)
        p_exp = df[CONFIG.tariff.export_price_column].to_numpy(dtype=float)

        soc = float(soc_init_kwh)

        charge_kwh = np.zeros(T)
        discharge_kwh = np.zeros(T)
        soc_kwh = np.zeros(T)
        grid_import_kwh = np.zeros(T)
        grid_export_kwh = np.zeros(T)

        for t in range(T):
            net = solar[t] - load[t]  # +ve => surplus solar, -ve => deficit

            if net > 0:
                room = max(0.0, self.battery.capacity_kwh - soc)

                max_charge_input = min(
                    e_max_charge,
                    room / self.battery.charge_efficiency
                    if self.battery.charge_efficiency > 0
                    else 0.0,
                )

                c = min(net, max_charge_input)
                charge_kwh[t] = c
                soc += self.battery.charge_efficiency * c
                net -= c

                if net > 0:
                    grid_export_kwh[t] = net

            elif net < 0:
                deficit = -net

                max_discharge_output = min(
                    e_max_discharge,
                    soc * self.battery.discharge_efficiency,
                )

                d = min(deficit, max_discharge_output)
                discharge_kwh[t] = d
                soc -= (
                    d / self.battery.discharge_efficiency
                    if self.battery.discharge_efficiency > 0
                    else 0.0
                )
                deficit -= d

                if deficit > 0:
                    grid_import_kwh[t] = deficit

            soc = max(self.battery.min_soc_kwh, min(self.battery.capacity_kwh, soc))
            soc_kwh[t] = soc

        out = pd.DataFrame(index=df.index)
        out["load_kwh"] = load
        out["solar_kwh"] = solar
        out["import_price_p_per_kwh"] = p_imp
        out["export_price_p_per_kwh"] = p_exp

        out["charge_kwh"] = charge_kwh
        out["discharge_kwh"] = discharge_kwh
        out["soc_kwh"] = soc_kwh
        out["grid_import_kwh"] = grid_import_kwh
        out["grid_export_kwh"] = grid_export_kwh

        out["cost_pence"] = (
            out["grid_import_kwh"] * out["import_price_p_per_kwh"]
            - out["grid_export_kwh"] * out["export_price_p_per_kwh"]
        )

        cost_pence = float(out["cost_pence"].sum())
        soc_end_kwh = float(out["soc_kwh"].iloc[-1])

        balance = (
            out["solar_kwh"]
            + out["grid_import_kwh"]
            + out["discharge_kwh"]
            - (
                out["load_kwh"]
                + out["charge_kwh"]
                + out["grid_export_kwh"]
            )
        )

        if np.abs(balance).max() > 1e-6:
            raise AssertionError(
                f"Energy balance violated in greedy "
                f"(max abs {np.abs(balance).max():.6f} kWh)"
            )

        return out, cost_pence, soc_end_kwh