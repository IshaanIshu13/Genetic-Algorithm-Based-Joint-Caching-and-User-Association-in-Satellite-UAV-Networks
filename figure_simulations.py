import numpy as np
from main_simulation import main_simulation_function


def run_average_simulation(N_value, iterations=50):
    total_costs = []
    for _ in range(iterations):
        cost = main_simulation_function(N=N_value)  # Pass N instead of Nu
        total_costs.append(cost)
    avg_cost = np.mean(total_costs)
    print(f"N={N_value}, Average Service Cost over {iterations} iterations = {avg_cost:.2f}")
    return avg_cost


if __name__ == "__main__":
    N_values = [500, 600, 700, 800, 900, 1000]  # example user counts you want to test
    avg_costs = {}
    for n in N_values:
        avg_costs[n] = run_average_simulation(n)
