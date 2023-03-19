# Firefly Algorithm

This repository contains Python implementations of the Cuckoo Search and Firefly Algorithm optimization algorithms. Both algorithms are nature-inspired metaheuristic algorithms that are particularly well-suited to optimization problems with complex search spaces.

## Firefly Algorithm

Firefly Algorithm is another optimization algorithm that is inspired by the behavior of natural organisms. In this case, the algorithm simulates the flashing behavior of fireflies to find the optimal solution to a given problem.

To use the Firefly Algorithm, import the `firefly_algorithm` function from the `firefly_algorithm.py` file and call it with your fitness function and the bounds of your search space:

```python
from firefly_algorithm import firefly_algorithm

def my_fitness_function(x):
    # Compute the fitness of the candidate solution x
    ...

best_solution, best_fitness = firefly_algorithm(my_fitness_function, dim=10, lb=-5, ub=5)
```

## License

This code is released under the [MIT License](LICENSE).
