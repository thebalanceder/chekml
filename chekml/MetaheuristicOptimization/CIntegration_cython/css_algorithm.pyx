# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

# Define numpy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class ChargedSystemSearch:
    cdef object objective_function
    cdef int dim
    cdef np.ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef double ka
    cdef double kv
    cdef double a
    cdef double epsilon
    cdef np.ndarray charged_particles
    cdef np.ndarray velocities
    cdef np.ndarray best_solution
    cdef double best_value
    cdef list charged_memory
    cdef int cm_size
    cdef list history

    def __init__(self, object objective_function, int dim, bounds, int population_size=16, 
                 int max_iter=3000, double ka=1.0, double kv=1.0, double a=1.0, double epsilon=1e-10):
        """
        Initialize the Charged System Search (CSS) optimizer.

        Parameters:
        - objective_function: Function to optimize (e.g., ECBI function for damage detection).
        - dim: Number of dimensions (variables, e.g., number of structural elements).
        - bounds: Tuple of (lower, high) bounds for each dimension.
        - population_size: Number of charged particles (CPs).
        - max_iter: Maximum number of iterations.
        - ka: Acceleration coefficient.
        - kv: Velocity coefficient.
        - a: Distance threshold for force calculation.
        - epsilon: Small positive number to avoid division by zero.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.ka = ka
        self.kv = kv
        self.a = a
        self.epsilon = epsilon
        self.charged_particles = None
        self.velocities = None
        self.best_solution = None
        self.best_value = np.inf
        self.charged_memory = []
        self.cm_size = population_size // 4
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_charged_particles(self):
        """Generate initial charged particles and velocities randomly."""
        self.charged_particles = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                                  (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim), dtype=DTYPE)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_charged_particles(self):
        """Compute fitness values for the charged particles."""
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.charged_particles[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calculate_charge(self, np.ndarray[DTYPE_t, ndim=1] fitness):
        """Calculate charge magnitude for each charged particle."""
        cdef double fitworst = np.max(fitness)
        cdef double fitbest = np.min(fitness)
        cdef np.ndarray[DTYPE_t, ndim=1] charges = np.ones(self.population_size, dtype=DTYPE)
        cdef int i
        if fitbest != fitworst:
            for i in range(self.population_size):
                charges[i] = (fitness[i] - fitworst) / (fitbest - fitworst)
        return charges

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def calculate_forces(self, np.ndarray[DTYPE_t, ndim=1] fitness):
        """Calculate resultant forces on each charged particle."""
        cdef np.ndarray[DTYPE_t, ndim=1] charges = self.calculate_charge(fitness)
        cdef np.ndarray[DTYPE_t, ndim=2] forces = np.zeros((self.population_size, self.dim), dtype=DTYPE)
        cdef int i, j, k
        cdef double r_ij, r_ij_norm, force_term, p_ij
        cdef np.ndarray[DTYPE_t, ndim=1] best_pos = (self.best_solution if self.best_solution is not None 
                                                     else self.charged_particles[0])

        for j in range(self.population_size):
            for i in range(self.population_size):
                if i != j:
                    # Calculate separation distance
                    r_ij = 0.0
                    for k in range(self.dim):
                        r_ij += (self.charged_particles[i, k] - self.charged_particles[j, k]) ** 2
                    r_ij = sqrt(r_ij)

                    # Normalized distance
                    r_ij_norm = 0.0
                    for k in range(self.dim):
                        r_ij_norm += ((self.charged_particles[i, k] + self.charged_particles[j, k]) / 2 - best_pos[k]) ** 2
                    r_ij_norm = r_ij / (sqrt(r_ij_norm) + self.epsilon)

                    # Probability of attraction
                    p_ij = 1.0 if (fitness[i] < fitness[j] or 
                                   (fitness[i] - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + self.epsilon) > np.random.rand()) else 0.0

                    # Force calculation based on distance
                    if r_ij < self.a:
                        force_term = (charges[i] / (self.a ** 3)) * r_ij
                    else:
                        force_term = charges[i] / (r_ij ** 2)

                    for k in range(self.dim):
                        forces[j, k] += p_ij * force_term * (self.charged_particles[i, k] - self.charged_particles[j, k]) * charges[j]

        return forces

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_positions(self, np.ndarray[DTYPE_t, ndim=2] forces):
        """Update positions and velocities of charged particles."""
        cdef double dt = 1.0
        cdef int i, k
        cdef double rand1, rand2
        for i in range(self.population_size):
            rand1 = np.random.rand()
            rand2 = np.random.rand()
            for k in range(self.dim):
                # Update velocity
                self.velocities[i, k] = rand1 * self.kv * self.velocities[i, k] + rand2 * self.ka * forces[i, k]
                # Update position
                self.charged_particles[i, k] += self.velocities[i, k] * dt
                # Boundary handling
                if self.charged_particles[i, k] < self.bounds[k, 0]:
                    self.charged_particles[i, k] = self.bounds[k, 0]
                elif self.charged_particles[i, k] > self.bounds[k, 1]:
                    self.charged_particles[i, k] = self.bounds[k, 1]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_charged_memory(self, np.ndarray[DTYPE_t, ndim=1] fitness):
        """Update charged memory with best solutions."""
        cdef np.ndarray[long, ndim=1] sorted_indices = np.argsort(fitness)
        cdef int i, j
        cdef double cm_worst_fitness
        for i in range(min(self.cm_size, len(sorted_indices))):
            if len(self.charged_memory) < self.cm_size:
                self.charged_memory.append((self.charged_particles[sorted_indices[i]].copy(), fitness[sorted_indices[i]]))
            else:
                cm_worst_fitness = max([cm[1] for cm in self.charged_memory])
                for j in range(len(self.charged_memory)):
                    if self.charged_memory[j][1] == cm_worst_fitness:
                        if fitness[sorted_indices[i]] < cm_worst_fitness:
                            self.charged_memory[j] = (self.charged_particles[sorted_indices[i]].copy(), fitness[sorted_indices[i]])
                        break

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Charged System Search optimization."""
        self.initialize_charged_particles()
        cdef np.ndarray[DTYPE_t, ndim=1] fitness
        cdef np.ndarray[DTYPE_t, ndim=2] forces
        cdef int iteration, min_idx
        
        for iteration in range(self.max_iter):
            fitness = self.evaluate_charged_particles()
            min_idx = np.argmin(fitness)
            
            # Update best solution
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.charged_particles[min_idx].copy()
                self.best_value = fitness[min_idx]
            
            # Calculate forces and update positions
            forces = self.calculate_forces(fitness)
            self.update_positions(forces)
            
            # Update charged memory
            self.update_charged_memory(fitness)
            
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")
        
        return self.best_solution, self.best_value, self.history

# Example usage for structural damage detection
if __name__ == "__main__":
    def ecbi_objective_function(X):
        """
        Example ECBI objective function for structural damage detection.
        X: Damage variables (reduction in stiffness for each element).
        Returns: ECBI value to be minimized.
        """
        return np.sum(X**2)  # Dummy objective function

    # Define problem parameters
    dim = 10  # Changed from cdef int dim to regular Python int
    bounds = [(0, 1)] * dim
    css = ChargedSystemSearch(ecbi_objective_function, dim, bounds, 
                              population_size=16, max_iter=3000)
    
    # Run optimization
    best_solution, best_value, history = css.optimize()
    print(f"Best Solution: {best_solution}")
    print(f"Best Value: {best_value}")
