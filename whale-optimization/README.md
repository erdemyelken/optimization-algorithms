# Whale Optimization Algorithm

This is a Python implementation of the Whale Optimization Algorithm (WOA) for solving optimization problems. WOA is a metaheuristic algorithm inspired by the hunting behavior of humpback whales.

## How to Use

To use this implementation, you need to create an instance of the WhaleOptimization class and pass in the necessary parameters. Here's an example:

```python
from whale_optimization import WhaleOptimization

# Define the search space and cost function
search_space = [-10, 10]
def cost_function(x):
    return sum(x**2)

# Create an instance of the WhaleOptimization class
woa = WhaleOptimization(search_space, cost_function, n_agents=10, max_iter=100)

# Run the optimization
best_solution = woa.optimize()

print("Best solution:", best_solution)

In this example, we're optimizing the cost_function, which is simply the sum of the squares of the input values. We're using a search space of [-10, 10], 10 agents, and 100 iterations. After running the optimization, we print out the best solution found.
## Parameters
The `WhaleOptimization` class takes the following parameters:

- `search_space` (list): The search space of the problem.
- `cost_function` (function): The cost function of the problem.
- `n_agents` (int, optional): The number of agents (default=5).
- `a_max` (float, optional): The maximum value of the `a` parameter (default=2).
- `a_min` (float, optional): The minimum value of the `a` parameter (default=0).
- `b_max` (float, optional): The maximum value of the `b` parameter (default=1).
- `b_min` (float, optional): The minimum value of the `b` parameter (default=0).
- `max_iter` (int, optional): The maximum number of iterations (default=100).

## Visualization
The `WhaleOptimization` class also provides a visualization method `visualize()` that creates an animated plot of the optimization process. Here's an example:

```python
woa = WhaleOptimization(search_space, cost_function, n_agents=10, max_iter=100)
woa.visualize()
## References
Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). The whale optimization algorithm. Advances in Engineering Software, 95, 51-67.
