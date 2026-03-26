from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd

from config import CONFIG, BatteryConfig


class BatteryScheduler(ABC):
    def __init__(self, battery_config: BatteryConfig | None = None):
        self.battery = battery_config or CONFIG.battery

    @abstractmethod
    def schedule_day(
        self,
        day_df: pd.DataFrame,
        soc_init_kwh: float,
    ) -> tuple[pd.DataFrame, float, float]:
        """
        Returns:
            results_df, cost_pence, soc_end_kwh
        """
        raise NotImplementedError