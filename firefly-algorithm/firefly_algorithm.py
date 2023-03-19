import numpy as np

def firefly_algorithm(fitness_func, dim, lb, ub, n=25, alpha=0.2, beta0=1, gamma=1.0, generations=100):
    # Initialize the population of fireflies
    fireflies = np.random.rand(n, dim) * (ub - lb) + lb

    # Evaluate the fitness of the initial fireflies
    fitness = fitness_func(fireflies)

    # Run the main Firefly Algorithm loop for the specified number of generations
    for generation in range(generations):
        # Compute attractiveness between fireflies
        beta = beta0 * np.exp(-gamma * generation)
        for i in range(len(fireflies)):
            for j in range(len(fireflies)):
                if fitness[i] > fitness[j]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta_i_j = beta / (r ** 2 + 1e-10)
                    fireflies[i] += beta_i_j * (fireflies[j] - fireflies[i])
        
        # Move the fireflies randomly to increase exploration
        step_size = alpha * (ub - lb)
        fireflies += step_size * (np.random.rand(*fireflies.shape) - 0.5)

        # Enforce bounds on firefly positions
        fireflies = np.maximum(fireflies, lb)
        fireflies = np.minimum(fireflies, ub)

        # Evaluate the fitness of the updated fireflies
        fitness = fitness_func(fireflies)

    # Get the best firefly and its fitness score
    best_idx = np.argmin(fitness)
    best_firefly = fireflies[best_idx]
    best_fitness = fitness[best_idx]

    return best_firefly, best_fitness
