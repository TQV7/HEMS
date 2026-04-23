from config import CONFIG
from data import build_merged_df
from simulate import simulate_period
from models.greedy import GreedyScheduler
from models.milp import MilpScheduler

df = build_merged_df()

greedy_scheduler = GreedyScheduler(CONFIG.battery)
milp_scheduler = MilpScheduler(CONFIG.battery)

greedy_ts, greedy_daily = simulate_period(
    merged=df,
    start_date="2024-01-01",
    end_date="2024-01-07",
    scheduler=greedy_scheduler,
    soc_init_kwh=CONFIG.battery.initial_soc_kwh,
)

milp_ts, milp_daily = simulate_period(
    merged=df,
    start_date="2024-01-01",
    end_date="2024-01-07",
    scheduler=milp_scheduler,
    soc_init_kwh=CONFIG.battery.initial_soc_kwh,
)

print("Greedy total cost (pence):", greedy_daily["cost_pence"].sum())
print("MILP total cost (pence):", milp_daily["cost_pence"].sum())
print()
print(greedy_daily.head())
print()
print(milp_daily.head())