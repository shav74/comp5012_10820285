# Multi-Objective Optimisation for the Periodic Travelling Salesman Problem (PTSP)

This project implements a custom multi-objective evolutionary optimiser for the Periodic Travelling Salesman Problem. The goal is to simultaneously minimise the total travel distance and balance daily workload over a configurable planning period.

## Features
- Custom flexible solution representation (day-to-customers mapping)
- Multiple mutation operators and crossover mechanisms
- Pareto-based selection without external diversity preservation
- Dynamic planning horizon (default: 60 days)
- Visual outputs: 
  - Pareto front plots
  - Fitness progression over generations
  - Customer visit heatmaps
  - Daily route distance bar charts
  - Route maps for selected solutions

## Technologies
- Python 3.12
- Matplotlib
- Seaborn
- Pandas
- NumPy
- (Optional) Anaconda environment for management

## How to Run
1. Clone this repository.
2. Install required libraries:
    ```bash
    pip install matplotlib seaborn pandas
    ```
3. Run the main optimiser script:
    ```bash
    python main.py
    ```
4. Outputs (visualisations and CSV files) will be saved in the project directory.
