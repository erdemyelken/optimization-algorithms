import random

def genetic_algorithm(fitness_func, pop_size, chromosome_length, mutation_rate, crossover_rate, max_iter):
    """
    Implements the Genetic Algorithm optimization algorithm.

    Args:
        fitness_func: The function to be optimized.
        pop_size: The population size.
        chromosome_length: The length of the chromosome.
        mutation_rate: The mutation rate.
        crossover_rate: The crossover rate.
        max_iter: The maximum number of iterations.

    Returns:
        The best individual and the best fitness value.
    """
    
    # Initialize the population
    population = [[random.randint(0, 1) for _ in range(chromosome_length)] for _ in range(pop_size)]
    
    for i in range(max_iter):
        # Evaluate the fitness of each individual
        fitness_values = [fitness_func(individual) for individual in population]
        
        # Select the parents for the next generation
        parents = []
        for _ in range(pop_size):
            # Select two individuals from the population at random
            ind1 = population[random.randint(0, pop_size-1)]
            ind2 = population[random.randint(0, pop_size-1)]
            
            # Choose the better individual as a parent
            if fitness_func(ind1) > fitness_func(ind2):
                parents.append(ind1)
            else:
                parents.append(ind2)
        
        # Create the next generation
        new_population = []
        for _ in range(pop_size):
            # Select two parents at random
            parent1 = parents[random.randint(0, pop_size-1)]
            parent2 = parents[random.randint(0, pop_size-1)]
            
            # Crossover
            if random.random() < crossover_rate:
                crossover_point = random.randint(1, chromosome_length-1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
            else:
                child1 = parent1
                child2 = parent2
            
            # Mutation
            if random.random() < mutation_rate:
                mutation_point = random.randint(0, chromosome_length-1)
                child1[mutation_point] = 1 - child1[mutation_point]
                child2[mutation_point] = 1 - child2[mutation_point]
            
            new_population.append(child1)
            new_population.append(child2)
        
        population = new_population
    
    # Find the best individual
    best_individual = max(population, key=fitness_func)
    best_fitness = fitness_func(best_individual)
    
    return best_individual, best_fitness
