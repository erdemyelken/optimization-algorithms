import random
from rosenbrock import rosenbrock

def fitness_func(individual):
    # Calculate the fitness value of the individual using rosenbrock function
    return 1 / (rosenbrock(individual) + 1)

def main():
    # Set the parameters for the genetic algorithm
    pop_size = 50
    chromosome_length = 10
    mutation_rate = 0.01
    crossover_rate = 0.8
    max_iter = 100
    
    # Run the genetic algorithm
    best_individual, best_fitness = genetic_algorithm(fitness_func, pop_size, chromosome_length, mutation_rate, crossover_rate, max_iter)
    
    print("Best individual: {}".format(best_individual))
    print("Best fitness: {}".format(best_fitness))

if __name__ == "__main__":
    main()
