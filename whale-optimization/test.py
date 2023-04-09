import numpy as np
from whale_optimization import WhaleOptimization

def sphere(x):
    """
    Sphere function.

    Parameters:
    x (np.array): Input array.

    Returns:
    float: Output value.
    """
    return sum(x**2)

if __name__ == "__main__":
    # Define search space
    search_space = [-10, 10]

    # Initialize algorithm
    wo = WhaleOptimization(search_space=search_space, cost_function=sphere, n_agents=10)

    # Run optimization
    wo.optimize()

    # Print results
    print("Best solution:", wo.global_best["position"])
    print("Best cost:", wo.global_best["cost"])
