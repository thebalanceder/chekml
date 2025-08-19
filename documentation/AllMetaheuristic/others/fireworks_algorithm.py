import numpy as np
from scipy.stats import poisson

class FireworksAlgorithm:
    def __init__(self, objective_function, dim, bounds, num_particles=50, max_iter=100, 
                 alpha=0.1, beta=1.0, delta_t=1.0):
        """
        Initialize the Fireworks Algorithm (FWA) optimizer.

        Parameters:
        - objective_function: Function to optimize (e.g., Sphere function).
        - dim: Number of dimensions (variables).
        - bounds: Tuple or array of (lower, upper) bounds for each dimension.
        - num_particles: Number of particles (fireworks).
        - max_iter: Maximum number of iterations (generations).
        - alpha: Explosion amplitude factor.
        - beta: Explosion magnitude scaling factor for Poisson distribution.
        - delta_t: Time step for spark generation.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.delta_t = delta_t

        self.particles = None  # Population of particles (fireworks)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_particles(self):
        """Generate initial particle positions randomly within bounds."""
        self.particles = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                          (self.num_particles, self.dim))

    def evaluate_fitness(self, particles=None):
        """Compute fitness values for the particles or specified array."""
        if particles is None:
            particles = self.particles
        return np.array([self.objective_function(p) for p in particles])

    def generate_sparks(self, best_particle):
        """Generate sparks around the best particle."""
        num_sparks = poisson.rvs(self.beta)  # Poisson distribution for number of sparks
        sparks = np.tile(best_particle, (num_sparks, 1)) + \
                 self.alpha * np.random.randn(num_sparks, self.dim) * self.delta_t
        return sparks

    def update_particles(self):
        """Update particle positions by combining with sparks and applying bounds."""
        # Evaluate current particles' fitness
        fitness = self.evaluate_fitness()
        min_idx = np.argmin(fitness)
        best_particle = self.particles[min_idx].copy()
        best_fitness = fitness[min_idx]

        # Update best solution if improved
        if best_fitness < self.best_value:
            self.best_solution = best_particle.copy()
            self.best_value = best_fitness

        # Generate sparks around the best particle
        sparks = self.generate_sparks(best_particle)

        # Concatenate particles and sparks
        all_particles = np.vstack((self.particles, sparks))

        # Apply boundary conditions
        all_particles = np.clip(all_particles, self.bounds[:, 0], self.bounds[:, 1])

        # Sort particles based on fitness
        fitness = self.evaluate_fitness(all_particles)
        sorted_indices = np.argsort(fitness)
        all_particles = all_particles[sorted_indices]

        # Select the top num_particles
        self.particles = all_particles[:self.num_particles]

    def optimize(self):
        """Run the Fireworks Algorithm optimization."""
        self.initialize_particles()
        for generation in range(self.max_iter):
            self.update_particles()
            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")
        return self.best_solution, self.best_value, self.history
