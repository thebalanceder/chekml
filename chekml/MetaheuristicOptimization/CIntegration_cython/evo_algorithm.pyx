# cython: language_level=3
cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt

# Define types for NumPy arrays
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class EnergyValleyOptimizer:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        double step_size
        double momentum
        double learning_rate
        np.ndarray particles
        np.ndarray velocity
        np.ndarray best_solution
        double best_value
        list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100,
                 double step_size=0.1, double momentum=0.9, double learning_rate=0.2):
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
        self.bounds = np.array(bounds, dtype=np.float64)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.particles = None
        self.velocity = None
        self.best_solution = None
        self.best_value = np.inf
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_particles(self):
        """ Randomly initialize particle positions and velocities """
        cdef double[:, :] bounds = self.bounds
        cdef double[:, :] particles = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                                       (self.population_size, self.dim))
        cdef double[:, :] velocity = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                                      (self.population_size, self.dim))
        self.particles = np.asarray(particles)
        self.velocity = np.asarray(velocity)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_particles(self):
        """ Compute fitness values for all particles """
        cdef double[:] fitness = np.zeros(self.population_size, dtype=np.float64)
        cdef double[:, :] particles = self.particles
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(particles[i])
        return np.asarray(fitness)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_gradient(self):
        """ Approximate gradient using finite differences """
        cdef double[:, :] gradient = np.zeros((self.population_size, self.dim), dtype=np.float64)
        cdef double[:, :] particles = self.particles
        cdef double[:] x_plus = np.zeros(self.dim, dtype=np.float64)
        cdef double[:] x_minus = np.zeros(self.dim, dtype=np.float64)
        cdef int i, j, k
        for i in range(self.population_size):
            for j in range(self.dim):
                # Copy particle
                for k in range(self.dim):
                    x_plus[k] = particles[i, k]
                    x_minus[k] = particles[i, k]
                # Perturb dimension j
                x_plus[j] += self.learning_rate
                x_minus[j] -= self.learning_rate
                gradient[i, j] = (self.objective_function(x_plus) -
                                 self.objective_function(x_minus)) / (2 * self.learning_rate)
        return np.asarray(gradient)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Energy Valley Optimization algorithm """
        self.initialize_particles()
        cdef double[:, :] particles = self.particles
        cdef double[:, :] velocity = self.velocity
        cdef double[:, :] bounds = self.bounds
        cdef double[:] fitness
        cdef np.ndarray sorted_indices
        cdef double[:] best_solution
        cdef double best_value
        cdef int iteration, min_idx, i, j

        for iteration in range(self.max_iter):
            # Evaluate fitness
            fitness = self.evaluate_particles()

            # Update best solution
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = np.asarray(particles[min_idx]).copy()
                self.best_value = fitness[min_idx]

            # Sort particles based on fitness
            sorted_indices = np.argsort(fitness)
            # Convert memoryviews to NumPy arrays for indexing
            particles_np = np.asarray(particles)
            fitness_np = np.asarray(fitness)
            particles_np = particles_np[sorted_indices]
            fitness_np = fitness_np[sorted_indices]
            # Update memoryviews by reassigning
            particles = particles_np
            fitness = fitness_np
            # Update the stored particles array
            self.particles = particles_np

            # Calculate gradient
            gradient = self.compute_gradient()

            # Update velocities and positions
            for i in range(self.population_size):
                for j in range(self.dim):
                    velocity[i, j] = self.momentum * velocity[i, j] + self.step_size * gradient[i, j]
                    particles[i, j] = particles[i, j] - velocity[i, j]

            # Ensure particles stay within bounds
            for i in range(self.population_size):
                for j in range(self.dim):
                    if particles[i, j] < bounds[j, 0]:
                        particles[i, j] = bounds[j, 0]
                    elif particles[i, j] > bounds[j, 1]:
                        particles[i, j] = bounds[j, 1]

            # Store history
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Fitness = {self.best_value}")

        return self.best_solution, self.best_value, self.history

# Example objective function (sphere function)
def sphere_function(x):
    """ Sphere function: sum of squares """
    return np.sum(np.array(x)**2)
