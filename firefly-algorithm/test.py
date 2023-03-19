from firefly_algorithm import firefly_algorithm
import numpy as np

# Define the fitness function
def sphere(x):
    return np.sum(x ** 2)

# Define the search space bounds
lb = -5.12
ub = 5.12

# Set the algorithm parameters
num_fireflies = 50
max_generation = 100
alpha = 0.5
beta0 = 1
gamma = 0.01

# Run the firefly algorithm to minimize the sphere function
best_sol, best_fitness = firefly_algorithm(sphere, lb, ub, num_fireflies, max_generation, alpha, beta0, gamma)

# Print the results
print('Best solution:', best_sol)
print('Best fitness:', best_fitness)
