from cuckoo_search import cuckoo_search
import numpy as np

# Define the fitness function to minimize
def fitness(x):
    return np.sum(x**2)

# Set the bounds of the search space
lb = -5.12
ub = 5.12

# Set the dimension of the search space
dim = 10

# Run the Cuckoo Search algorithm
best_nest, best_fitness = cuckoo_search(fitness, dim, lb, ub)

# Print the results
print("Best solution found:")
print(best_nest)
print("Best fitness found:")
print(best_fitness)
