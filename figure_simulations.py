import numpy as np
from main_simulation import main_simulation_function

def run_average_simulation(Nu_value, iterations=50):
    total_costs = []
    for _ in range(iterations):
        cost = main_simulation_function(Nu=Nu_value)
        total_costs.append(cost)
    avg_cost = np.mean(total_costs)
    print(f"Nu={Nu_value}, Average Service Cost over {iterations} iterations = {avg_cost:.2f}")
    return avg_cost

if __name__ == "__main__":
    Nu_values = [10, 13, 16, 19, 22]
    avg_costs = {}
    for nu in Nu_values:
        avg_costs[nu] = run_average_simulation(nu)
