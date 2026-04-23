from __future__ import annotations

import numpy as np
import pandas as pd
from pulp import (
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
)

from config import CONFIG, BatteryConfig
from .scheduler_base import BatteryScheduler


class MilpExportAllowedScheduler(BatteryScheduler):
    """
    Mixed-integer linear programming battery scheduler.

    Decision variables per half-hour slot:
      - charge_kwh
      - discharge_kwh
      - soc_kwh
      - grid_import_kwh
      - grid_export_kwh
      - is_charging (binary)
      - is_importing (binary)

    The binary mode variables prevent:
      - simultaneous charging and discharging
      - simultaneous grid import and export
    """

    def __init__(
        self,
        battery_config: BatteryConfig | None = None,
        solver_msg: bool | None = None,
    ):
        super().__init__(battery_config=battery_config)
        self.solver_msg = CONFIG.solver.msg if solver_msg is None else solver_msg

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
        import_prices = df[CONFIG.tariff.import_price_column].to_numpy(dtype=float)
        export_prices = df[CONFIG.tariff.export_price_column].to_numpy(dtype=float)

        model = LpProblem("BatteryScheduleDayMILP", LpMinimize)

        charge = [LpVariable(f"charge_{t}", lowBound=0) for t in range(T)]
        discharge = [LpVariable(f"discharge_{t}", lowBound=0) for t in range(T)]
        soc = [
            LpVariable(
                f"soc_{t}",
                lowBound=self.battery.min_soc_kwh,
                upBound=self.battery.capacity_kwh,
            )
            for t in range(T)
        ]
        grid_import = [LpVariable(f"grid_import_{t}", lowBound=0) for t in range(T)]
        grid_export = [LpVariable(f"grid_export_{t}", lowBound=0) for t in range(T)]

        # Battery mode:
        # 1 => charging allowed, discharge forced to 0
        # 0 => discharging allowed, charge forced to 0
        is_charging = [LpVariable(f"is_charging_{t}", cat=LpBinary) for t in range(T)]

        # Grid mode:
        # 1 => importing allowed, export forced to 0
        # 0 => exporting allowed, import forced to 0
        is_importing = [LpVariable(f"is_importing_{t}", cat=LpBinary) for t in range(T)]

        model += (
            soc[0]
            == soc_init_kwh
            + self.battery.charge_efficiency * charge[0]
            - discharge[0] / self.battery.discharge_efficiency
        )

        for t in range(1, T):
            model += (
                soc[t]
                == soc[t - 1]
                + self.battery.charge_efficiency * charge[t]
                - discharge[t] / self.battery.discharge_efficiency
            )

        for t in range(T):
            model += charge[t] <= e_max_charge * is_charging[t]
            model += discharge[t] <= e_max_discharge * (1 - is_charging[t])

        grid_big_m = (
            float(np.max(load))
            + float(np.max(solar))
            + e_max_charge
            + e_max_discharge
            + self.battery.capacity_kwh
        )

        for t in range(T):
            model += grid_import[t] <= grid_big_m * is_importing[t]
            model += grid_export[t] <= grid_big_m * (1 - is_importing[t])


        for t in range(T):
            model += (
                solar[t] + grid_import[t] + discharge[t]
                == load[t] + charge[t] + grid_export[t]
            )


        if CONFIG.simulation.enforce_terminal_soc_floor:
            model += soc[T - 1] >= CONFIG.simulation.terminal_soc_floor_kwh


        throughput_penalty = CONFIG.simulation.throughput_penalty_per_kwh

        model += lpSum(
            grid_import[t] * import_prices[t]
            - grid_export[t] * export_prices[t]
            + throughput_penalty * (charge[t] + discharge[t])
            for t in range(T)
        )


        solver = PULP_CBC_CMD(
            msg=self.solver_msg,
            timeLimit=CONFIG.solver.time_limit_seconds,
        )
        status = model.solve(solver)
        status_str = LpStatus[status]

        if status_str != "Optimal":
            raise RuntimeError(f"Solver did not find optimal solution: {status_str}")


        out = pd.DataFrame(index=df.index)
        out["load_kwh"] = load
        out["solar_kwh"] = solar
        out["import_price_p_per_kwh"] = import_prices
        out["export_price_p_per_kwh"] = export_prices

        out["charge_kwh"] = [float(v.value() or 0.0) for v in charge]
        out["discharge_kwh"] = [float(v.value() or 0.0) for v in discharge]
        out["soc_kwh"] = [float(v.value() or 0.0) for v in soc]
        out["grid_import_kwh"] = [float(v.value() or 0.0) for v in grid_import]
        out["grid_export_kwh"] = [float(v.value() or 0.0) for v in grid_export]
        out["is_charging"] = [int(round(float(v.value() or 0.0))) for v in is_charging]
        out["is_importing"] = [int(round(float(v.value() or 0.0))) for v in is_importing]

        out["cost_pence"] = (
            out["grid_import_kwh"] * out["import_price_p_per_kwh"]
            - out["grid_export_kwh"] * out["export_price_p_per_kwh"]
        )

        if throughput_penalty != 0:
            out["throughput_penalty_pence"] = throughput_penalty * (
                out["charge_kwh"] + out["discharge_kwh"]
            )
            out["objective_contribution_pence"] = (
                out["cost_pence"] + out["throughput_penalty_pence"]
            )

        cost_pence = float(out["cost_pence"].sum())
        soc_end_kwh = float(out["soc_kwh"].iloc[-1])

        # Sanity checks
        tol = 1e-6

        if (out["soc_kwh"] < self.battery.min_soc_kwh - tol).any():
            raise AssertionError("SOC fell below minimum bound")
        if (out["soc_kwh"] > self.battery.capacity_kwh + tol).any():
            raise AssertionError("SOC exceeded capacity bound")

        if (out["charge_kwh"] < -tol).any() or (out["discharge_kwh"] < -tol).any():
            raise AssertionError("Negative charge/discharge encountered")

        if (out["charge_kwh"] > e_max_charge + tol).any():
            raise AssertionError("Charge exceeded per-slot limit")
        if (out["discharge_kwh"] > e_max_discharge + tol).any():
            raise AssertionError("Discharge exceeded per-slot limit")

        simultaneous_battery = (out["charge_kwh"] > tol) & (out["discharge_kwh"] > tol)
        if simultaneous_battery.any():
            raise AssertionError("Simultaneous charge and discharge detected")

        simultaneous_grid = (out["grid_import_kwh"] > tol) & (out["grid_export_kwh"] > tol)
        if simultaneous_grid.any():
            raise AssertionError("Simultaneous grid import and export detected")

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
        if np.abs(balance).max() > 1e-4:
            raise AssertionError(
                f"Energy balance violated (max abs {np.abs(balance).max():.6f} kWh)"
            )

        return out, cost_pence, soc_end_kwh