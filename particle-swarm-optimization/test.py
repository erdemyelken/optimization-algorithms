import numpy as np
from pso import PSO

def test_function(x):
    return (x[0] - 3) ** 2 + (x[1] - 4) ** 2 + 5

def main():
    num_dimensions = 2
    swarm_size = 10
    num_iterations = 100
    lower_bound = np.array([-10, -10])
    upper_bound = np.array([10, 10])

    pso = PSO(num_dimensions, swarm_size, num_iterations, lower_bound, upper_bound, test_function)
    gbest_pos, gbest_val = pso.run()

    print("Global best position:", gbest_pos)
    print("Global best value:", gbest_val)

if __name__ == "__main__":
    main()
