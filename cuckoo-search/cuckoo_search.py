import numpy as np

def generate_levy_flight(beta):
    # Generate a random step from a Levy distribution
    # with parameter beta
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
             (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(*x0.shape) * sigma
    v = np.random.randn(*x0.shape)
    step = u / np.abs(v) ** (1 / beta)
    return step

def get_cuckoo_nests(n, d, lb, ub):
    # Initialize cuckoo nests randomly within the search space
    nests = np.random.rand(n, d) * (ub - lb) + lb
    return nests

def get_best_nest(nests, fitness_func):
    # Get the nest with the best fitness score
    fitness = fitness_func(nests)
    best_idx = np.argmin(fitness)
    return nests[best_idx], fitness[best_idx]

def levy_flight(nests, pa, beta):
    # Update the nests using a Levy flight strategy
    for i in range(len(nests)):
        step_size = pa * generate_levy_flight(beta)
        step = step_size * (nests[i] - nests[np.random.randint(len(nests))])
        nests[i] += step
    return nests

def replace_worst_nests(nests, fitness_func, lb, ub, pa):
    # Replace the worst nests with new randomly initialized ones
    fitness = fitness_func(nests)
    idxs = np.argsort(fitness)
    worst_nests = nests[idxs[-pa:]]
    new_nests = get_cuckoo_nests(pa, nests.shape[1], lb, ub)
    nests[idxs[-pa:]] = new_nests
    return nests

def cuckoo_search(fitness_func, dim, lb, ub, n=25, pa=0.25, beta=1.5, generations=100):
    # Initialize the population of cuckoo nests
    nests = get_cuckoo_nests(n, dim, lb, ub)

    # Evaluate the fitness of the initial nests
    best_nest, best_fitness = get_best_nest(nests, fitness_func)

    # Run the main Cuckoo Search loop for the specified number of generations
    for generation in range(generations):
        # Update the nests using a Levy flight strategy
        nests = levy_flight(nests, pa, beta)

        # Replace the worst nests with new randomly initialized ones
        nests = replace_worst_nests(nests, fitness_func, lb, ub, pa)

        # Keep track of the best nest found so far
        current_best_nest, current_best_fitness = get_best_nest(nests, fitness_func)
        if current_best_fitness < best_fitness:
            best_nest, best_fitness = current_best_nest, current_best_fitness

    return best_nest, best_fitness
