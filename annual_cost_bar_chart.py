from pathlib import Path
import matplotlib.pyplot as plt

# Data
strategies = ["Greedy baseline", "MILP (no export)", "MILP (export allowed)"]
annual_costs = [370, 172, 91]

# Create output folder if needed
output_dir = Path("outputs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot
plt.figure(figsize=(8, 5))
bars = plt.bar(strategies, annual_costs)

# Labels and title
plt.ylabel("Annual electricity cost (£)")
plt.title("Annual electricity cost by scheduling strategy")

# Add value labels above bars
for bar, value in zip(bars, annual_costs):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 5,
        f"£{value}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

# Save and show
save_path = output_dir / "annual_cost_comparison_bar_chart.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved chart to: {save_path}")