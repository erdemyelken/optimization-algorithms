import numpy as np

class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1])
        self.velocity = np.random.uniform(-1, 1)
        self.best_position = self.position

    def update_velocity(self, global_best_position, omega, phip, phig):
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        cognitive_velocity = phip * r1 * (self.best_position - self.position)
        social_velocity = phig * r2 * (global_best_position - self.position)
        self.velocity = omega * self.velocity + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        self.position += self.velocity
        if self.position < bounds[0]:
            self.position = bounds[0]
            self.velocity = 0
        elif self.position > bounds[1]:
            self.position = bounds[1]
            self.velocity = 0

class PSO:
    def __init__(self, objective_function, bounds, num_particles, max_iterations, omega, phip, phig):
        self.objective_function = objective_function
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.omega = omega
        self.phip = phip
        self.phig = phig
        self.particles = []
        self.global_best_position = None
        self.global_best_value = np.inf

    def initialize_particles(self):
        for i in range(self.num_particles):
            particle = Particle(self.bounds)
            self.particles.append(particle)

    def run(self):
        self.initialize_particles()

        for iteration in range(self.max_iterations):
            for particle in self.particles:
                fitness = self.objective_function(particle.position)
                if fitness < self.global_best_value:
                    self.global_best_position = particle.position
                    self.global_best_value = fitness
                if fitness < self.objective_function(particle.best_position):
                    particle.best_position = particle.position
                particle.update_velocity(self.global_best_position, self.omega, self.phip, self.phig)
                particle.update_position(self.bounds)

        return self.global_best_position, self.global_best_value
