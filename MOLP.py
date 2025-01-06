# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 20:44:22 2024

#########################################################
Objectives:
1 Maximize Line Utilization: We want to increase the production efficiency 
    (denoted as f1 = x1 + x2)
2 Minimize Lead Time: Reduce the time to fulfill orders 
    (denoted as f2 = 0.2x1 + 0.5x2)
3 Minimize Carbon Emissions: Minimize carbon footprint in production and 
    transportation (denoted as f3 = 0.4*x1 + 0.2*x2)
#########################################################

#########################################################
decision variables
x1: Units produced on line A.
𝑥2: Units produced on line B.
#########################################################

#########################################################
constraints
Capacity Constraints: Line A can produce a maximum of 100 units, and line B can 
    produce a maximum of 120 units.
0≤𝑥1≤100 and 0≤𝑥2≤120

Demand Constraints: The total demand is at least 150 units.
𝑥1 + 𝑥2 ≥ 150

Line Utilization Constraint: To ensure balanced use of both lines, let’s say 
    the utilization of line B 
    should be at least 50% of line A’s utilization.
x2 ≥ 0.5x1
​#########################################################
"""

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# Define the supply chain multi-objective problem
class SupplyChainProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2,  # Two decision variables (production on two lines)
                         n_obj=3,  # Three objectives (line utilization, lead time, carbon emissions)
                         n_constr=2,  # two constraints
                         xl=np.array([0.0, 0.0]),  # Lower bounds
                         xu=np.array([100.0, 120.0]))  # Upper bounds

    def _evaluate(self, x, out, *args, **kwargs):
        # Objectives
        f1 = -(x[:, 0] + x[:, 1])  # Maximize line utilization (negated for minimization)
        f2 = 0.2 * x[:, 0] + 0.5 * x[:, 1]  # Minimize lead time
        f3 = 0.4 * x[:, 0] + 0.2 * x[:, 1]  # Minimize carbon emissions
        
        # Constraints
        g1 = 150 - (x[:, 0] + x[:, 1])  # Demand constraint: x1 + x2 >= 150
        g2 = 0.5 * x[:, 0] - x[:, 1]  # Line utilization balance: x2 >= 0.5 * x1
        
        out["F"] = np.column_stack([f1, f2, f3])
        out["G"] = np.column_stack([g1, g2])

# Instantiate the supply chain problem
problem = SupplyChainProblem()

# NSGA-II algorithm
algorithm = NSGA2(pop_size=100)

# Run optimization
res = minimize(problem, algorithm, ('n_gen', 200), seed=1, verbose=False)

# Get feasible solutions
feasible_solutions = res.F[np.all(res.G <= 0, axis=1)]

Feasible_Solutions = pd.DataFrame(feasible_solutions)
Feasible_Solutions = Feasible_Solutions.set_axis(['Obj_f1','Obj_f2','Obj_f3'], axis=1)


# Plot feasible and infeasible areas
plot = Scatter(title="Pareto Front with Feasibility")
plot.add(feasible_solutions, color="green", label="Feasible")
infeasible_solutions = res.F[np.any(res.G > 0, axis=1)]
plot.add(infeasible_solutions, color="red", label="Infeasible")
plot.show()

# Compute the Ideal and Nadir points
ideal_point = np.min(feasible_solutions, axis=0)
nadir_point = np.max(feasible_solutions, axis=0)

print(f"Ideal point: {ideal_point}")
print(f"Nadir point: {nadir_point}")

# Visualize Pareto front with Ideal and Nadir points
plot = Scatter(title="Pareto Front with Ideal and Nadir Points")
plot.add(feasible_solutions, color="green", label="Feasible")
plot.add(ideal_point.reshape(1, -1), color="blue", s=100, marker="*", label="Ideal Point")
plot.add(nadir_point.reshape(1, -1), color="orange", s=100, marker="*", label="Nadir Point")
plot.show()

# Print the first few Pareto-optimal solutions
print("First 5 Pareto-optimal solutions:")
print(feasible_solutions[:5])

# Extract each objective from the feasible solutions
line_utilization = -feasible_solutions[:, 0]  # Negate to interpret correctly
lead_time = feasible_solutions[:, 1]
carbon_emissions = feasible_solutions[:, 2]

# Analyze the first solution
print(f"Solution 1: Line Utilization = {line_utilization[0]}, Lead Time = {lead_time[0]}, Carbon Emissions = {carbon_emissions[0]}")

# Plot the trade-offs (if you want to visualize it further)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(lead_time, carbon_emissions, c=line_utilization, cmap="viridis", label="Pareto Solutions")
plt.colorbar(label="Line Utilization")
plt.xlabel("Lead Time")
plt.ylabel("Carbon Emissions")
plt.title("Trade-off between Lead Time and Carbon Emissions")
plt.show()

"""
#########################################################
A business case

Balancing Cost and Lead Time in Supplier Selection
Problem: A company needs to select suppliers for a product, balancing two key 
objectives: 
    minimizing cost and 
    minimizing lead time. 
    
The goal is to find an optimal mix of suppliers that achieves a reasonable 
    trade-off between these objectives.

Solution Approach
We'll solve this as a MOO problem using weighted sum or Pareto optimization to 
balance the objectives of cost and lead time.


We define a custom problem SupplierSelectionProblem with two objectives: 
    minimizing cost and minimizing lead time.
    
The _evaluate function calculates the cost and lead time based on supplier selection, 
where x represents a binary decision (0 or 1) for each supplier.

We use NSGA2, a popular genetic algorithm for multi-objective optimization, 
to find Pareto-optimal solutions across generations.

The solutions show different trade-offs, helping the decision-maker choose a 
balanced approach.
#########################################################
"""

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import numpy as np
import matplotlib.pyplot as plt

# Define the supply chain optimization problem
class SupplierSelectionProblem(Problem):
    def __init__(self, num_suppliers=3):
        super().__init__(n_var=num_suppliers, n_obj=2, xl=0, xu=1)
        self.costs = np.array([200, 300, 250])  # Example supplier costs
        self.lead_times = np.array([5, 8, 6])  # Lead times in days

    def _evaluate(self, x, out, *args, **kwargs):
        # Calculate cost and lead time for selected suppliers
        total_cost = np.dot(x, self.costs)
        total_lead_time = np.dot(x, self.lead_times)

        # Objective values: minimize both cost and lead time
        out["F"] = [total_cost, total_lead_time]

# Instantiate problem and algorithm
problem = SupplierSelectionProblem()
algorithm = NSGA2(pop_size=100)

# Solve the problem
result = minimize(problem, algorithm, termination=('n_gen', 50))


# Show results
for i in range(len(result.F)):
    print(f"Solution {i+1}: Cost = {result.F[i][0]}, Lead Time = {result.F[i][1]}, Suppliers Selected = {result.X[i]}")

"""
Non-Dominated Solutions: NSGA-II generates a set of optimal solutions known as 
the Pareto front. Each solution represents a different trade-off between the objectives 
(e.g., cost vs. lead time). 

You can access additional solutions by iterating through result.F and result.X arrays, 
which store multiple solutions.
"""
for i in range(len(result.F)):
    print(f"Solution {i+1}: Cost = {result.F[i][0]}, Lead Time = {result.F[i][1]}, Suppliers Selected = {result.X[i]}")

#Viz
"""
This scatter plot will show you the spread of the Pareto-optimal solutions, 
helping you understand how cost and lead time trade-offs vary.
"""
costs = result.F[:, 0]
lead_times = result.F[:, 1]

plt.scatter(costs, lead_times, c='blue', label='Pareto Front')
plt.xlabel("Cost")
plt.ylabel("Lead Time")
plt.title("Pareto Front for Supplier Selection Problem")
plt.legend()
plt.show()


"""
#########################################################
A complex Multi-Objective Optimization problem

Problem Statement: You work for a global company with a network of suppliers, 
manufacturers, and distribution centers. Your goal is to optimize the supplier 
selection and order quantity allocation to balance three main objectives

objectives:
    1 Minimizing Total Cost: This includes raw material costs, transportation, 
    and inventory holding costs.
    
    2 Minimizing Carbon Emissions: Carbon footprint varies by supplier based on 
    distance and transport type.
    
    3 Maximizing Resilience: This is a measure of the risk associated with each 
    supplier, considering factors like reliability, geopolitical risks, and 
    supplier capacity flexibility.

Each supplier has constraints on maximum and minimum order quantities. 
Additionally, there is a requirement to ensure that a minimum percentage of total 
materials comes from low-risk suppliers to improve supply chain resilience.

Solution Approach:

Define Objectives:
    Objective 1: Minimize total cost.
    Objective 2: Minimize total carbon emissions.
    Objective 3: Maximize resilience (treated as maximizing a supplier risk metric).

Constraints:
    1 Minimum order quantity from each selected supplier.
    2 Maximum order quantity due to supplier capacity.
    3 Requirement that at least 30% of total materials come from low-risk 
    suppliers.
    
    4.a total orders should rannge between a and b
    4.b total orders should be exactly a
    
Python Implementation: This setup would use a library like pymoo or DEAP for 
multi-objective optimization, setting up constraints and objectives accordingly.

#########################################################
"""

# Libraries
# --------------------------------
#for MOO
from pymoo.core.problem import Problem #for MOO
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

#for numeric and table manuplation
import numpy as np 
import pandas as pd

#for Viz
import matplotlib.pyplot as plt 
# --------------------------------


# Setup example data
# --------------------------------

#total number of suppliers
num_suppliers = 5

#oder qty constraint 4.a and 4.b
a = 750
b = 1000

#transportation CO2e from suppliers
carbon_footprints = np.array([0.5, 0.3, 0.4, 0.6, 0.2])

#risk factor of each supplier
risks = np.array([0.1, 0.5, 0.2, 0.4, 0.3])

#MOQ for each supplier
min_orders = np.array([100, 50, 150, 100, 80])

#Max capacity for each supplier
max_orders = np.array([500, 300, 400, 350, 200])

#unit cost of each supplier
cost_per_unit = np.array([10, 15, 12, 8, 20])
# --------------------------------


# --------------------------------
# Define Multi-Objective Optimization Problem
# --------------------------------
class SupplyChainMOO(Problem):
    def __init__(self, num_suppliers, carbon_footprints, risks, min_orders, max_orders, cost_per_unit):
        super().__init__(n_var=num_suppliers, n_obj=3, n_constr=1, xl=min_orders, xu=max_orders)
        self.carbon_footprints = carbon_footprints
        self.risks = risks
        self.cost_per_unit = cost_per_unit

    def _evaluate(self, X, out, *args, **kwargs):
        # X will be a (pop_size, num_suppliers) array, i.e., multiple solutions evaluated at once.
        # Define arrays to store each objective for all solutions in X.
        
        # Objective 1: Total Cost
        cost = np.sum(X * self.cost_per_unit, axis=1)  # shape: (pop_size,)
        
        # Objective 2: Carbon Emissions
        emissions = np.sum(X * self.carbon_footprints, axis=1)  # shape: (pop_size,)
        
        # Objective 3: Resilience (minimize risk, so multiply by -1 to maximize resilience)
        resilience = -1 * np.sum(X * self.risks, axis=1)  # shape: (pop_size,)
        
        # Constraint 1: at least 30% from low-risk suppliers (assuming risk below 0.3 is low)
        low_risk_constraint = 0.3 * np.sum(X, axis=1) - np.sum(X * (self.risks < 0.3), axis=1)  # shape: (pop_size,)

        
        # Assign objectives and constraints to the output dictionary
        out["F"] = np.column_stack([cost, emissions, resilience])  # shape: (pop_size, n_obj)
        out["G"] = low_risk_constraint # shape: (pop_size,)



class SupplyChainMOO_4a(Problem):
    def __init__(self, num_suppliers, carbon_footprints, risks, min_orders, max_orders, cost_per_unit):
        super().__init__(n_var=num_suppliers, n_obj=3, n_constr=3, xl=min_orders, xu=max_orders)  # n_constr=3
        self.carbon_footprints = carbon_footprints
        self.risks = risks
        self.cost_per_unit = cost_per_unit

    def _evaluate(self, X, out, *args, **kwargs):
        # Objective 1: Total Cost
        cost = np.sum(X * self.cost_per_unit, axis=1)
        
        # Objective 2: Carbon Emissions
        emissions = np.sum(X * self.carbon_footprints, axis=1)
        
        # Objective 3: Resilience (minimize risk, so multiply by -1 to maximize resilience)
        resilience = -1 * np.sum(X * self.risks, axis=1)
        
        # Constraint 1: at least 30% of orders from low-risk suppliers
        low_risk_constraint = 0.3 * np.sum(X, axis=1) - np.sum(X * (self.risks < 0.3), axis=1)
        
        # Constraint 2: Total orders should be at least 700
        min_total_order_constraint = a - np.sum(X, axis=1)
        
        # Constraint 3: Total orders should not exceed 1000
        max_total_order_constraint = np.sum(X, axis=1) - b
        
        # Assign objectives and constraints to the output dictionary
        out["F"] = np.column_stack([cost, emissions, resilience])
        out["G"] = np.column_stack([low_risk_constraint, min_total_order_constraint, max_total_order_constraint])


class SupplyChainMOO_4b(Problem):
    def __init__(self, num_suppliers, carbon_footprints, risks, min_orders, max_orders, cost_per_unit):
        super().__init__(n_var=num_suppliers, n_obj=3, n_constr=3, xl=min_orders, xu=max_orders)  # n_constr=3
        self.carbon_footprints = carbon_footprints
        self.risks = risks
        self.cost_per_unit = cost_per_unit

    def _evaluate(self, X, out, *args, **kwargs):
        # Objective 1: Total Cost
        cost = np.sum(X * self.cost_per_unit, axis=1)
        
        # Objective 2: Carbon Emissions
        emissions = np.sum(X * self.carbon_footprints, axis=1)
        
        # Objective 3: Resilience (minimize risk, so multiply by -1 to maximize resilience)
        resilience = -1 * np.sum(X * self.risks, axis=1)
        
        # Constraint 1: at least 30% of orders from low-risk suppliers
        low_risk_constraint = 0.3 * np.sum(X, axis=1) - np.sum(X * (self.risks < 0.3), axis=1)
        
        # Constraint 2: Total orders should be at least 750
        min_total_order_constraint = a - np.sum(X, axis=1)
        
        # Constraint 3: Total orders should not exceed 750
        max_total_order_constraint = np.sum(X, axis=1) - b
        
        # Assign objectives and constraints to the output dictionary
        out["F"] = np.column_stack([cost, emissions, resilience])
        out["G"] = np.column_stack([low_risk_constraint, min_total_order_constraint, max_total_order_constraint])

# --------------------------------
        
# --------------------------------
#Solver
# --------------------------------
algorithm = NSGA2(pop_size=100)  # You can set a larger population for more diverse solutions

# Instantiate problem and algorithm
problem = SupplyChainMOO(num_suppliers, carbon_footprints, risks, min_orders, max_orders, cost_per_unit)
problem_4a = SupplyChainMOO_4a(num_suppliers, carbon_footprints, risks, min_orders, max_orders, cost_per_unit)
problem_4b = SupplyChainMOO_4b(num_suppliers, carbon_footprints, risks, min_orders, max_orders, cost_per_unit)

# Solve the problem
result = minimize(problem, algorithm, termination=('n_gen', 50), verbose=False)
result_4a = minimize(problem_4a, algorithm, termination=('n_gen', 50), verbose=False)
result_4b = minimize(problem_4b, algorithm, termination=('n_gen', 50), verbose=False)
# --------------------------------


# --------------------------------
# Add Decision Variables and Results into a DF (contraint 4 not explored, variables min_total_order_constraint, max_total_order_constraint)
# --------------------------------
Supplier_1 = []
Supplier_2 = []
Supplier_3 = []
Supplier_4 = []
Supplier_5 = []

Results_cost = []
Results_emissions = []
Results_Resilience = []
Results_Risk = []

#Access the Decision Variables
# obtain allocations for each supplier from Pareto-optimal solution
for i, sol in enumerate(result.X):
    Supplier_1.append(sol[0])
    Supplier_2.append(sol[1])
    Supplier_3.append(sol[2])
    Supplier_4.append(sol[3])
    Supplier_5.append(sol[4])
        
#access Pareto-front solutions     
for solution in result.F:
    Results_cost.append(solution[0])
    Results_emissions.append(solution[1])
    Results_Resilience.append(-solution[2]) # Reverse resilience to positive value

for solution in result.G:
    Results_Risk.append(-solution[0])

ResultsDB = pd.DataFrame({'OrderQ_Supp_1':Supplier_1,
                          'OrderQ_Supp_2':Supplier_2,
                          'OrderQ_Supp_3':Supplier_3,
                          'OrderQ_Supp_4':Supplier_4,
                          'OrderQ_Supp_5':Supplier_5,
                          'Cost':Results_cost, 
                          'Emissions':Results_emissions, 
                          'Resilience':Results_Resilience,
                          'Risk':Results_Risk})

ResultsDB[['OrderQ_Supp_1','OrderQ_Supp_2','OrderQ_Supp_3', 'OrderQ_Supp_4',  'OrderQ_Supp_5']].agg('sum', axis=1)
# --------------------------------
   
# --------------------------------
# Visualization of Pareto Front (Cost vs Emissions)
# --------------------------------
plt.scatter(x = Results_cost, y = Results_emissions, c= Results_Resilience, s = Results_Risk, cmap='viridis')
plt.colorbar(label='Resilience')
plt.xlabel('Cost')
plt.ylabel('Emissions')
plt.title('Pareto Front of Multi-Objective Optimization')
plt.show()
# --------------------------------

# --------------------------------
#Interpretation for Stakeholders (Technical Level)
# --------------------------------

"""
The multi-objective optimization (MOO) results give us several solutions 
that balance our objectives—minimizing cost, emissions, and risk. 

This approach doesn’t produce a single “best” solution but a Pareto front, 
which is a set of solutions where improving one objective means compromising on 
another.

Each solution on this front represents a different trade-off:
    Cost: Total expenditure based on order quantities and supplier prices.
    Emissions: Total CO₂e footprint, which helps assess environmental impact.
    Resilience: This is related to supplier risk; higher resilience (lower risk) 
        usually means choosing suppliers with lower risk factors, though this 
        may increase cost or emissions.


Example Stakeholder Summary:
We can present each solution as a "scenario" where, for example, Scenario A minimizes 
emissions but is costlier, while Scenario B lowers cost but slightly raises emissions.
Visualizing these solutions on a Pareto chart (e.g., Cost vs. Emissions with color 
indicating Resilience) can help stakeholders quickly grasp the trade-offs and 
select the solution that best aligns with company priorities 
(e.g., reducing environmental impact, cost-effectiveness, or risk mitigation).
"""
