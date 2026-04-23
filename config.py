from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 10.0
    max_charge_kw: float = 3.0
    max_discharge_kw: float = 3.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    min_soc_kwh: float = 0.0
    initial_soc_kwh: float = 5.0
    timestep_hours: float = 0.5

    @property
    def max_charge_kwh_per_slot(self) -> float:
        return self.max_charge_kw * self.timestep_hours

    @property
    def max_discharge_kwh_per_slot(self) -> float:
        return self.max_discharge_kw * self.timestep_hours


# Simulation settings
@dataclass(frozen=True)
class SimulationConfig:
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    timestep_minutes: int = 30

    use_forecasts: bool = False
    load_forecast_lookback_days: int = 7

    enforce_terminal_soc_floor: bool = False
    terminal_soc_floor_kwh: float = 0.0

    throughput_penalty_per_kwh: float = 0.01


# Tariff settings
@dataclass(frozen=True)
class TariffConfig:
    use_fixed_export_price: bool = True
    fixed_export_price_p_per_kwh: float = 15.0

    import_price_column: str = "import_price_p_per_kwh"
    export_price_column: str = "export_price_p_per_kwh"


# Solver settings
@dataclass(frozen=True)
class SolverConfig:
    solver_name: str = "PULP_CBC_CMD"
    msg: bool = False
    time_limit_seconds: int | None = None


# Data settings
@dataclass(frozen=True)
class DataConfig:
    load_filename: str = "house-load-2024.csv"
    price_filename: str = "agile-prices-2024.csv"
    solar_filename: str = "solar_data.py"

    timestamp_column: str = "timestamp"
    load_column: str = "load_kwh"
    solar_column: str = "solar_kwh"

    split_hourly_solar_into_half_hours: bool = True


# Plot settings
@dataclass(frozen=True)
class PlotConfig:
    save_figures: bool = True
    show_figures: bool = True
    dpi: int = 150


# Scenario definition
@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    battery: BatteryConfig


# Top-level app config
@dataclass(frozen=True)
class AppConfig:
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    tariff: TariffConfig = field(default_factory=TariffConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    data: DataConfig = field(default_factory=DataConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)

    scenarios: tuple[ScenarioConfig, ...] = field(
        default_factory=lambda: (
            ScenarioConfig(
                name="battery_5kwh",
                battery=BatteryConfig(capacity_kwh=5.0, initial_soc_kwh=2.5),
            ),
            ScenarioConfig(
                name="battery_10kwh",
                battery=BatteryConfig(capacity_kwh=10.0, initial_soc_kwh=5.0),
            ),
            ScenarioConfig(
                name="battery_15kwh",
                battery=BatteryConfig(capacity_kwh=15.0, initial_soc_kwh=7.5),
            ),
        )
    )


CONFIG = AppConfig()