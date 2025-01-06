Multi-Objective Optimization for Supply Chain

This repository demonstrates multi-objective optimization (MOO) techniques applied to supply chain problems. 
The focus is on balancing key objectives like cost, emissions, and resilience, with constraints like supplier capacity and order demands.

Objectives:
    1. Maximize Line Utilization: Increase production efficiency.
    2. Minimize Lead Time: Reduce order fulfillment time.
    3. Minimize Carbon Emissions: Decrease environmental impact.

Features:
1. Optimization Models:
    Single Tier: Balances production efficiency, lead time, and emissions across production lines.
    Multi-Tier Supply Chain: Optimizes supplier selection and order allocations.
    Resilience Integration: Considers risk factors in decision-making.
2. Algorithms: Built with NSGA-II (Non-dominated Sorting Genetic Algorithm II) for Pareto front optimization.
3. Visualizations: Heatmaps and Pareto front charts to highlight trade-offs

Prerequisites:
1. Tools: Python 3.8+
2. Libraries: See requirements.txt

Installation:
Clone the repository: git clone https://github.com/your-username/supply-chain-moo.git
Navigate to the directory: cd supply-chain-moo
Install dependencies: pip install -r requirements.txt

Usage:
1. Run the main scripts:
    For Multi-Tier Allocation: python MOLP.py
2. Visualize outputs:
    2.1 Pareto fronts showing trade-offs between objectives.
    2.2 Heatmaps for allocations across suppliers or production lines.


Repository Contents:
1. MOLP.py: Demonstrates multi-objective linear programming (MOLP) for various supply chain scenarios.
2. Pareto Visualizations: Highlights trade-offs between objectives like cost, emissions, and resilience.
3. Feasible Solutions: Data saved for further analysis or reporting.

Visualization Example:
The repository includes visualizations such as:
1. Pareto fronts for cost, emissions, and resilience.
2. Heatmaps for allocation between suppliers and production lines.


Contributions: Contributions are welcome. Feel free to open an issue or create a pull request to enhance the models.

License: Licensed under the MIT License.

