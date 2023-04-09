# Particle Swarm Optimization (PSO) Algorithm

This repository contains an implementation of the Particle Swarm Optimization (PSO) algorithm in Python. The PSO algorithm is a metaheuristic optimization algorithm that is commonly used to solve optimization problems.

## pso.py

The `pso.py` file contains the implementation of the PSO algorithm. The PSO class is defined in this file, which has the following methods:

* `__init__(self, obj_func, num_particles, dimensions, options)`: Initializes the PSO algorithm with the objective function, number of particles, dimensions, and options.
* `optimize(self)`: Runs the PSO algorithm and returns the best particle and its fitness value.

## test.py

The `test.py` file contains an example usage of the PSO algorithm. In this file, a test function (`sphere_function`) is defined, which is used as the objective function in the PSO algorithm. The PSO algorithm is then initialized with this objective function and run for 100 iterations. The best particle and its fitness value are printed at the end of the optimization process.

To run the test, simply execute the `test.py` file in the terminal:
```
python test.py
```
## Usage

To use the PSO algorithm in your own project, follow these steps:

1. Import the PSO class from the `pso.py` file:
`from pso import PSO`
2. Define your own objective function. The objective function should take a 1D numpy array as input and return a scalar fitness value. Here is an example of a sphere function:
```
def sphere_function(x):
return sum(x**2)
```
## Options

The PSO algorithm can be customized by passing options to the PSO class constructor. Here are the available options:

* `c1` (float): Cognitive parameter.
* `c2` (float): Social parameter.
* `w` (float): Inertia weight.
* `max_iterations` (int): Maximum number of iterations.
* `tolerance` (float): Tolerance for convergence.

## References

Kennedy, J. and Eberhart, R. (1995). Particle swarm optimization. In Proceedings of IEEE International Conference on Neural Networks, pages 1942-1948.

Eberhart, R. and Kennedy, J. (1995). A new optimizer using particle swarm theory. In Proceedings of the sixth international symposium on micro machine and human science, pages 39-43.
