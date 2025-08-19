# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport sqrt, log, cos, M_PI
from libc.time cimport time

# Define types
ctypedef np.double_t DTYPE_t

# Simple Gaussian random number generator (Box-Muller transform)
cdef double randn() nogil:
    cdef double u1 = (<double>rand() / RAND_MAX)
    cdef double u2 = (<double>rand() / RAND_MAX)
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2)

# Uniform random number generator [0, 1]
cdef double uniform() nogil:
    return <double>rand() / RAND_MAX

cdef class FireworksAlgorithm:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int num_particles
        int max_iter
        double alpha
        double beta
        double delta_t
        np.ndarray particles
        np.ndarray best_solution
        double best_value
        list history

    def __init__(self, object objective_function, int dim, object bounds, int num_particles=50, 
                 int max_iter=100, double alpha=0.1, double beta=1.0, double delta_t=1.0):
        """
        Initialize the Fireworks Algorithm (FWA) optimizer.

        Parameters:
        - objective_function: Function to optimize (e.g., Sphere function).
        - dim: Number of dimensions (variables).
        - bounds: Array of (lower, upper) bounds for each dimension.
        - num_particles: Number of particles (fireworks).
        - max_iter: Maximum number of iterations (generations).
        - alpha: Explosion amplitude factor.
        - beta: Explosion magnitude scaling factor (used as fixed spark count).
        - delta_t: Time step for spark generation.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.delta_t = delta_t
        self.best_value = float("inf")
        self.history = []

    cdef void initialize_particles(self):
        """Generate initial particle positions randomly within bounds."""
        cdef int i, j
        self.particles = np.empty((self.num_particles, self.dim), dtype=np.double)
        cdef double[:, :] particles = self.particles
        cdef double[:] lb = self.bounds[:, 0]
        cdef double[:] ub = self.bounds[:, 1]
        
        for i in range(self.num_particles):
            for j in range(self.dim):
                particles[i, j] = lb[j] + (ub[j] - lb[j]) * uniform()

    cdef double[:] evaluate_fitness(self, double[:, :] particles=None):
        """Compute fitness values for the particles or specified array."""
        if particles is None:
            particles = self.particles
        cdef int n = particles.shape[0]
        cdef double[:] fitness = np.empty(n, dtype=np.double)
        cdef int i
        for i in range(n):
            fitness[i] = self.objective_function(np.asarray(particles[i]))
        return fitness

    cdef double[:, :] generate_sparks(self, double[:] best_particle):
        """Generate sparks around the best particle."""
        # Use beta as fixed number of sparks (instead of Poisson)
        cdef int num_sparks = <int>self.beta
        cdef double[:, :] sparks = np.empty((num_sparks, self.dim), dtype=np.double)
        cdef int i, j
        for i in range(num_sparks):
            for j in range(self.dim):
                sparks[i, j] = best_particle[j] + self.alpha * randn() * self.delta_t
        return sparks

    cdef void update_particles(self):
        """Update particle positions by combining with sparks and applying bounds."""
        cdef double[:, :] particles = self.particles
        cdef double[:] fitness = self.evaluate_fitness(particles)
        cdef int min_idx = np.argmin(fitness)
        cdef double[:] best_particle = particles[min_idx]
        cdef double best_fitness = fitness[min_idx]

        # Update best solution if improved
        if best_fitness < self.best_value:
            self.best_solution = np.asarray(best_particle).copy()
            self.best_value = best_fitness

        # Generate sparks around the best particle
        cdef double[:, :] sparks = self.generate_sparks(best_particle)

        # Concatenate particles and sparks
        cdef double[:, :] all_particles = np.vstack((particles, sparks))
        cdef int n_total = all_particles.shape[0]

        # Apply boundary conditions
        cdef int i, j
        cdef double[:] lb = self.bounds[:, 0]
        cdef double[:] ub = self.bounds[:, 1]
        for i in range(n_total):
            for j in range(self.dim):
                if all_particles[i, j] < lb[j]:
                    all_particles[i, j] = lb[j]
                elif all_particles[i, j] > ub[j]:
                    all_particles[i, j] = ub[j]

        # Sort particles based on fitness
        cdef double[:] all_fitness = self.evaluate_fitness(all_particles)
        # Convert to NumPy array for sorting
        all_particles_np = np.asarray(all_particles)
        cdef np.ndarray[long] sorted_indices = np.argsort(all_fitness)
        all_particles_np = all_particles_np[sorted_indices]
        self.particles = all_particles_np[:self.num_particles]

    cpdef tuple optimize(self):
        """Run the Fireworks Algorithm optimization."""
        # Seed random number generator
        srand(<unsigned int>time(NULL))
        
        self.initialize_particles()
        cdef int generation
        for generation in range(self.max_iter):
            self.update_particles()
            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")
        return self.best_solution, self.best_value, self.history

# Default Sphere function for testing
def sphere_function(x):
    """Sphere function: sum(x^2)"""
    return np.sum(x**2)
