from data import build_merged_df
from models.milp import MilpScheduler
from config import CONFIG

df = build_merged_df()
day_df = df.iloc[:48]

scheduler = MilpScheduler(CONFIG.battery)

out, cost_pence, soc_end_kwh = scheduler.schedule_day(
    day_df=day_df,
    soc_init_kwh=CONFIG.battery.initial_soc_kwh,
)

print(out.head())
print()
print("cost_pence:", cost_pence)
print("soc_end_kwh:", soc_end_kwh)
print("simultaneous charge/discharge:",
      ((out["charge_kwh"] > 1e-6) & (out["discharge_kwh"] > 1e-6)).any())
print(out[["discharge_kwh", "grid_export_kwh", "grid_import_kwh", "soc_kwh"]].head(10))