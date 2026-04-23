# Home Energy Management System (HEMS) Optimisation

## Overview

This project develops a **Home Energy Management System (HEMS)** for optimally scheduling residential battery storage under **dynamic electricity tariffs**.

The system integrates:

* Household electricity demand
* Solar photovoltaic (PV) generation
* Battery energy storage
* Time-varying electricity import/export prices

A **Mixed-Integer Linear Programming (MILP)** model is used to determine optimal battery schedules and is compared against a **greedy rule-based baseline**.

The goal is to evaluate how optimisation improves household electricity cost under realistic UK tariff conditions.

---

## Key Features

* MILP-based optimisation using PuLP
* Greedy baseline controller for benchmarking
* Full-year simulation (half-hour resolution, 2024 data)
* Battery modelling with:

  * capacity constraints
  * efficiency losses
  * power limits
* Multiple operational scenarios:

  * battery export allowed vs restricted
  * terminal state-of-charge constraints
* Automated outputs:

  * cost comparisons
  * seasonal analysis
  * battery capacity sensitivity

---

## Results Summary

* ~**54% cost reduction** vs greedy baseline (no export)
* ~**75% cost reduction** when battery export is allowed
* Savings depend strongly on:

  * tariff volatility
  * battery capacity
  * seasonal demand and solar generation

---

## Repository Structure

```text
HEMS_Project/
├── main.py                     # Entry point for running simulations
├── simulate.py                 # Rolling daily simulation engine
├── config.py                   # System and battery configuration
├── data.py                     # Data loading and preprocessing
├── plots.py                    # Plotting utilities
├── experiment_log.txt          # Experiment notes/log
│
├── models/                     # Scheduling models
│   ├── scheduler_base.py       # Base scheduler interface
│   ├── greedy.py               # Greedy baseline scheduler
│   ├── milp.py                 # MILP (no export)
│   └── milpExportAllowed.py    # MILP (export allowed)
│
├── analysis/                   # Scripts for figures and analysis
│   ├── annual_cost_bar_chart.py
│   └── make_background_figures.py
│
├── data/                       # Input datasets
│   ├── agile-prices-2024.csv
│   ├── house-load-2024.csv
│   └── solar_data.py
│
├── outputs/                    # Generated outputs
│   ├── figures/                # Plots used in report
│   └── results/                # CSV simulation outputs
│
└── tests/                      # Unit tests
    ├── test_data.py
    ├── test_greedy.py
    ├── test_milp.py
    └── test_simulate.py
```

---

## Methodology

### Optimisation Model

The scheduling problem is formulated as a **Mixed-Integer Linear Program (MILP)**.

**Decision variables:**

* Battery charge / discharge
* Grid import / export
* Energy flow allocation

**Constraints:**

* Energy balance (demand must be met)
* Battery state-of-charge dynamics
* Charge/discharge mutual exclusivity
* Grid import/export exclusivity
* Physical limits (capacity, power, efficiency)

**Objective:**
Minimise total electricity cost:

[
\sum_t (import_t \cdot p_t^{imp} - export_t \cdot p_t^{exp} + \lambda (charge_t + discharge_t))
]

---

### Simulation Framework

* **Rolling day-ahead optimisation**
* Battery state carried across days
* Full-year simulation (2024 dataset)
* Identical inputs for all strategies → fair comparison

---

## Assumptions

* Perfect foresight of demand, solar generation, and prices
* Deterministic optimisation (no uncertainty)
* Simplified battery degradation via throughput penalty
* Single household model

These assumptions provide an **upper-bound estimate** of achievable performance.

---

## Installation

### Requirements

* Python 3.9+
* pip

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run full simulation

```bash
python main.py
```

### Outputs

Results are saved in:

* `outputs/results/`:

  * daily cost results
  * time-series energy flows
  * battery capacity experiments

* `outputs/figures/`:

  * cost comparison plots
  * seasonal profiles
  * battery behaviour visualisations

---

## Testing

Run all tests:

```bash
pytest tests/
```

Tests include:

* data loading and preprocessing
* greedy scheduler behaviour
* MILP constraint validation
* simulation correctness

---

## Limitations

* Perfect foresight assumption (not realistic in deployment)
* No detailed battery degradation modelling
* Single household dataset
* Computational cost may limit scalability

---

## Author

Trong Vu
Supervisor: Andrew Coles

