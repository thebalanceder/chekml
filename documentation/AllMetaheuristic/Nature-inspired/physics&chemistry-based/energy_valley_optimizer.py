import numpy as np

class EnergyValleyOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 step_size=0.1, momentum=0.9, learning_rate=0.2):
        """
        Initialize the Energy Valley Optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of particles (solutions).
        - max_iter: Maximum number of iterations.
        - step_size: Velocity coefficient.
        - momentum: Momentum coefficient.
        - learning_rate: Learning rate for gradient approximation.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.momentum = momentum
        self.learning_rate = learning_rate

        self.particles = None  # Population of particles (solutions)
        self.velocity = None   # Particle velocities
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_particles(self):
        """ Randomly initialize particle positions and velocities """
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        self.particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocity = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def evaluate_particles(self):
        """ Compute fitness values for all particles """
        return np.array([self.objective_function(particle) for particle in self.particles])

    def compute_gradient(self):
        """ Approximate gradient using finite differences """
        gradient = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            for j in range(self.dim):
                x_plus = self.particles[i].copy()
                x_minus = self.particles[i].copy()
                x_plus[j] += self.learning_rate
                x_minus[j] -= self.learning_rate
                gradient[i, j] = (self.objective_function(x_plus) - 
                                self.objective_function(x_minus)) / (2 * self.learning_rate)
        return gradient

    def optimize(self):
        """ Run the Energy Valley Optimization algorithm """
        self.initialize_particles()
        
        for iteration in range(self.max_iter):
            # Evaluate fitness for each particle
            fitness = self.evaluate_particles()
            
            # Update best solution
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.particles[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Sort particles based on fitness
            sorted_indices = np.argsort(fitness)
            self.particles = self.particles[sorted_indices]
            fitness = fitness[sorted_indices]

            # Calculate gradient
            gradient = self.compute_gradient()

            # Update velocities and positions
            self.velocity = self.momentum * self.velocity + self.step_size * gradient
            self.particles = self.particles - self.velocity

            # Ensure particles stay within bounds
            self.particles = np.clip(self.particles, self.bounds[:, 0], self.bounds[:, 1])

            # Store history
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Fitness = {self.best_value}")

        return self.best_solution, self.best_value, self.history

# Example objective function (sphere function)
def sphere_function(x):
    """ Sphere function: sum of squares """
    return np.sum(x**2)
