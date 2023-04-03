import numpy as np
from grey-wolf-optimizer import gwo_search

# Rosenbrock function
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# Define the lower and upper bounds of the variables to be optimized
lb = [-5, -5]
ub = [5, 5]

# Define the number of dimensions of the search space
dim = 2

# Define the number of search agents
SearchAgents_no = 5

# Define the maximum number of iterations
Max_iter = 100

# Run GWO algorithm to find the minimum of the Rosenbrock function
best_pos, best_score = gwo_search(rosenbrock, lb, ub, dim, SearchAgents_no, Max_iter)

# Print the results
print("Best position: ", best_pos)
print("Best score: ", best_score)
