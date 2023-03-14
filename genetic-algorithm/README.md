# Genetic Algorithm
This module contains the implementation of the Genetic Algorithm optimization algorithm.

# Usage
The genetic_algorithm() function implements the genetic algorithm. It takes the following parameters:

- fitness_func: The function to be optimized.
- pop_size: The population size.
- chromosome_length: The length of the chromosome.
- mutation_rate: The mutation rate.
- crossover_rate: The crossover rate.
- max_iter: The maximum number of iterations.
- The function returns the best individual and the best fitness value.

```
from genetic_algorithm import genetic_algorithm

# Define the fitness function to optimize
def fitness(x):
    return sum(x)

# Set the parameters
pop_size = 100
chromosome_length = 5
mutation_rate = 0.01
crossover_rate = 0.8
max_iter = 1000

# Run the genetic algorithm
best_individual, best_fitness = genetic_algorithm(fitness, pop_size, chromosome_length, mutation_rate, crossover_rate, max_iter)

# Print the results
print(f"Best individual: {best_individual}")
print(f"Best fitness: {best_fitness}")
```

# License
This module is licensed under the MIT License. See the LICENSE file for more information.
