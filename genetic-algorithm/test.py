from genetic_algorithm import genetic_algorithm

def test_genetic_algorithm():
    def fitness_func(chromosome):
        return sum(chromosome)

    pop_size = 50
    chromosome_length = 10
    mutation_rate = 0.01
    crossover_rate = 0.8
    max_iter = 100

    best_individual, best_fitness = genetic_algorithm(fitness_func, pop_size, chromosome_length, mutation_rate, crossover_rate, max_iter)

    print("Best individual: ", best_individual)
    print("Best fitness: ", best_fitness)

if __name__ == "__main__":
    test_genetic_algorithm()
