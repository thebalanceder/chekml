#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sin, sqrt, fabs
from cython cimport floating

cnp.import_array()

cdef class WaterwheelPlantOptimizer:
    cdef public:
        object objective_function
        int dim
        cnp.ndarray bounds
        int population_size
        int max_iter
        tuple r1_range
        tuple r2_range
        tuple r3_range
        double k_initial
        tuple f_range
        tuple c_range
        cnp.ndarray waterwheels
        cnp.ndarray best_solution
        double best_value
        list history
        cnp.ndarray stagnation_counts

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=500, 
                 r1_range=(0, 2), r2_range=(0, 1), r3_range=(0, 2), double k_initial=1.0, 
                 f_range=(-5, 5), c_range=(-5, 5)):
        """
        Initialize the Waterwheel Plant Algorithm (WWPA) optimizer.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of waterwheel plants (solutions).
        - max_iter: Maximum number of iterations.
        - r1_range: Range for random variable r1 in exploration phase.
        - r2_range: Range for random variable r2 in exploration phase.
        - r3_range: Range for random variable r3 in exploitation phase.
        - k_initial: Initial value for parameter K.
        - f_range: Range for random variable f in mutation.
        - c_range: Range for random variable c in mutation.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_iter = max_iter
        self.r1_range = r1_range
        self.r2_range = r2_range
        self.r3_range = r3_range
        self.k_initial = k_initial
        self.f_range = f_range
        self.c_range = c_range

        self.waterwheels = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []
        self.stagnation_counts = np.zeros(population_size, dtype=np.int32)

    cpdef void initialize_waterwheels(self):
        """Initialize waterwheel plant positions randomly within bounds."""
        cdef cnp.ndarray[cnp.float64_t, ndim=2] bounds = self.bounds
        self.waterwheels = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                            (self.population_size, self.dim))

    cpdef cnp.ndarray[cnp.float64_t, ndim=1] evaluate_waterwheels(self):
        """Compute fitness values for the waterwheel plant positions."""
        cdef cnp.ndarray[cnp.float64_t, ndim=1] fitness = np.empty(self.population_size, dtype=np.float64)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.waterwheels[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray[cnp.float64_t, ndim=1] exploration_phase(self, int index, int t, double K):
        """Simulate position identification and hunting of insects (exploration)."""
        cdef double r1 = np.random.uniform(self.r1_range[0], self.r1_range[1])
        cdef double r2 = np.random.uniform(self.r2_range[0], self.r2_range[1])
        cdef cnp.ndarray[cnp.float64_t, ndim=1] W = np.empty(self.dim, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] new_position = np.empty(self.dim, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] current_position = self.waterwheels[index]
        cdef cnp.ndarray[cnp.float64_t, ndim=2] bounds = self.bounds
        cdef int j
        cdef double new_value, current_value

        # Equation (4): W = r1 * (P(t) + 2K)
        for j in range(self.dim):
            W[j] = r1 * (current_position[j] + 2 * K)

        # Equation (5): P(t+1) = P(t) + W * (2K + r2)
        for j in range(self.dim):
            new_position[j] = current_position[j] + W[j] * (2 * K + r2)
            new_position[j] = min(max(new_position[j], bounds[j, 0]), bounds[j, 1])

        # Check for stagnation and apply mutation if needed (Equation 6)
        new_value = self.objective_function(new_position)
        current_value = self.objective_function(current_position)

        if new_value < current_value:
            self.stagnation_counts[index] = 0
            return new_position
        else:
            self.stagnation_counts[index] += 1
            if self.stagnation_counts[index] >= 3:
                # Equation (6): P(t+1) = Gaussian(mu_P, sigma) + r1 * ((P(t) + 2K) / W)
                mu_P = np.mean(self.waterwheels, axis=0)  # Moved outside conditional
                sigma = np.std(self.waterwheels, axis=0)
                gaussian_term = np.random.normal(mu_P, sigma)
                for j in range(self.dim):
                    if W[j] != 0:  # Avoid division by zero
                        new_position[j] = gaussian_term[j] + r1 * (current_position[j] + 2 * K) / W[j]
                    else:
                        new_position[j] = gaussian_term[j]
                    new_position[j] = min(max(new_position[j], bounds[j, 0]), bounds[j, 1])
                self.stagnation_counts[index] = 0
            return new_position

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray[cnp.float64_t, ndim=1] exploitation_phase(self, int index, int t, double K):
        """Simulate carrying the insect to the suitable tube (exploitation)."""
        cdef double r3 = np.random.uniform(self.r3_range[0], self.r3_range[1])
        cdef cnp.ndarray[cnp.float64_t, ndim=1] W = np.empty(self.dim, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] new_position = np.empty(self.dim, dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] current_position = self.waterwheels[index]
        cdef cnp.ndarray[cnp.float64_t, ndim=1] best_solution = self.best_solution
        cdef cnp.ndarray[cnp.float64_t, ndim=2] bounds = self.bounds
        cdef int j
        cdef double new_value, current_value

        # Equation (7): W = r3 * (K * P_best(t) + r3 * P(t))
        for j in range(self.dim):
            W[j] = r3 * (K * best_solution[j] + r3 * current_position[j])

        # Equation (8): P(t+1) = P(t) + K * W
        for j in range(self.dim):
            new_position[j] = current_position[j] + K * W[j]
            new_position[j] = min(max(new_position[j], bounds[j, 0]), bounds[j, 1])

        # Check for stagnation and apply mutation if needed (Equation 9)
        new_value = self.objective_function(new_position)
        current_value = self.objective_function(current_position)

        if new_value < current_value:
            self.stagnation_counts[index] = 0
            return new_position
        else:
            self.stagnation_counts[index] += 1
            if self.stagnation_counts[index] >= 3:
                # Equation (9): P(t+1) = (r1 + K) * sin((f / c) * theta)
                r1 = np.random.uniform(self.r1_range[0], self.r1_range[1])
                f = np.random.uniform(self.f_range[0], self.f_range[1])
                c = np.random.uniform(self.c_range[0], self.c_range[1])
                theta = np.random.rand() * 2 * np.pi
                for j in range(self.dim):
                    if c != 0:  # Avoid division by zero
                        new_position[j] = (r1 + K) * sin((f / c) * theta)
                    else:
                        new_position[j] = current_position[j]
                    new_position[j] = min(max(new_position[j], bounds[j, 0]), bounds[j, 1])
                self.stagnation_counts[index] = 0
            return new_position

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double update_k(self, int t, double f):
        """Update the K parameter exponentially (Equation 10)."""
        return 1 + (2 * t * t / (self.max_iter * self.max_iter)) + f

    cpdef tuple optimize(self):
        """Run the Waterwheel Plant Algorithm optimization process."""
        self.initialize_waterwheels()
        
        cdef int t, i, min_idx
        cdef cnp.ndarray[cnp.float64_t, ndim=1] fitness
        cdef double r, f, K

        for t in range(self.max_iter):
            fitness = self.evaluate_waterwheels()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.waterwheels[min_idx].copy()
                self.best_value = fitness[min_idx]
            
            # Update parameters
            r = np.random.rand()
            f = np.random.uniform(self.f_range[0], self.f_range[1])
            K = self.update_k(t, f)
            
            # Update each waterwheel's position
            for i in range(self.population_size):
                if r < 0.5:
                    # Exploration phase
                    self.waterwheels[i] = self.exploration_phase(i, t, K)
                else:
                    # Exploitation phase
                    self.waterwheels[i] = self.exploitation_phase(i, t, K)
            
            # Update history
            self.history.append((t, self.best_solution.copy(), self.best_value))
            print(f"Iteration {t + 1}: Best Value = {self.best_value}")
        
        return self.best_solution, self.best_value, self.history

# Example usage
if __name__ == "__main__":
    # Example objective function: Sphere function
    def sphere_function(x):
        return np.sum(x**2)
    
    # Define problem parameters
    dim = 30
    bounds = [(-100, 100)] * dim  # Bounds for each dimension
    population_size = 50
    max_iter = 500
    
    # Initialize and run optimizer
    optimizer = WaterwheelPlantOptimizer(
        objective_function=sphere_function,
        dim=dim,
        bounds=bounds,
        population_size=population_size,
        max_iter=max_iter
    )
    best_solution, best_value, history = optimizer.optimize()
    
    print(f"Best Solution: {best_solution}")
    print(f"Best Value: {best_value}")
