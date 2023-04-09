import numpy as np
from whale_optimization import WhaleOptimization

def rosenbrock(x):
    """
    Rosenbrock function.
    """
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# Define the search space
search_space = [-2, 2]

# Define the number of agents
n_agents = 30

# Define the maximum number of iterations
max_iter = 100

# Define the Whale Optimization Algorithm
woa = WhaleOptimization(search_space=search_space, cost_function=rosenbrock, n_agents=n_agents, max_iter=max_iter)

# Run the optimization algorithm
best_position, best_cost, cost_progress = woa.optimize()

# Print the result
print("Best position:", best_position)
print("Best cost:", best_cost)

# Plot the cost progress
import matplotlib.pyplot as plt
plt.plot(cost_progress)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Progress")
plt.show()
