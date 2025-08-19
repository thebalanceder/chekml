# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

# Define numpy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class GreyWolfOptimizer:
    cdef public:
        object objective_function
        int dim
        int population_size
        int max_iter
        np.ndarray bounds
        np.ndarray positions
        np.ndarray alpha_pos
        double alpha_score
        np.ndarray beta_pos
        double beta_score
        np.ndarray delta_pos
        double delta_score
        np.ndarray convergence_curve
        list alpha_history

    def __init__(self, objective_function, int dim, bounds, int population_size=30, int max_iter=500):
        """
        Initialize the Grey Wolf Optimizer (GWO).

        Parameters:
        - objective_function: Function to optimize (minimization problem).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension or single pair for all.
        - population_size: Number of search agents (wolves).
        - max_iter: Maximum number of iterations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        
        # Handle bounds
        if np.isscalar(bounds[0]):  # Single bound for all dimensions
            self.bounds = np.array([[bounds[0], bounds[1]] for _ in range(dim)], dtype=DTYPE)
        else:  # Different bounds for each dimension
            self.bounds = np.array(bounds, dtype=DTYPE)
        
        self.positions = None  # Population of search agents
        self.alpha_pos = np.zeros(dim, dtype=DTYPE)
        self.alpha_score = float("inf")
        self.beta_pos = np.zeros(dim, dtype=DTYPE)
        self.beta_score = float("inf")
        self.delta_pos = np.zeros(dim, dtype=DTYPE)
        self.delta_score = float("inf")
        self.convergence_curve = np.zeros(max_iter, dtype=DTYPE)
        self.alpha_history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_positions(self):
        """Initialize the positions of search agents."""
        cdef int i
        if len(self.bounds) == 1:  # Single bound for all dimensions
            self.positions = np.random.uniform(self.bounds[0, 0], self.bounds[0, 1],
                                             (self.population_size, self.dim)).astype(DTYPE)
        else:  # Different bounds for each dimension
            self.positions = np.zeros((self.population_size, self.dim), dtype=DTYPE)
            for i in range(self.dim):
                self.positions[:, i] = np.random.uniform(self.bounds[i, 0], self.bounds[i, 1],
                                                        self.population_size)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def enforce_bounds(self):
        """Return search agents that go beyond the boundaries to the search space."""
        cdef int i, j
        cdef double[:, :] positions = self.positions
        cdef double[:, :] bounds = self.bounds
        for i in range(self.population_size):
            for j in range(self.dim):
                positions[i, j] = np.clip(positions[i, j], bounds[j, 0], bounds[j, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_hierarchy(self):
        """Update Alpha, Beta, and Delta wolves based on fitness."""
        cdef int i
        cdef double fitness
        cdef double[:] alpha_pos = self.alpha_pos
        cdef double[:] beta_pos = self.beta_pos
        cdef double[:] delta_pos = self.delta_pos
        cdef double[:, :] positions = self.positions
        
        for i in range(self.population_size):
            fitness = self.objective_function(positions[i])
            if fitness < self.alpha_score:
                self.delta_score = self.beta_score
                delta_pos[:] = beta_pos
                self.beta_score = self.alpha_score
                beta_pos[:] = alpha_pos
                self.alpha_score = fitness
                alpha_pos[:] = positions[i]
            elif fitness < self.beta_score and fitness > self.alpha_score:
                self.delta_score = self.beta_score
                delta_pos[:] = beta_pos
                self.beta_score = fitness
                beta_pos[:] = positions[i]
            elif fitness < self.delta_score and fitness > self.beta_score:
                self.delta_score = fitness
                delta_pos[:] = positions[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_positions(self, int iteration):
        """Update the positions of search agents based on Alpha, Beta, and Delta."""
        cdef double a = 2 - iteration * (2.0 / self.max_iter)  # Linearly decrease a from 2 to 0
        cdef int i, j
        cdef double r1, r2, A1, C1, D_alpha, X1
        cdef double A2, C2, D_beta, X2
        cdef double A3, C3, D_delta, X3
        cdef double[:] alpha_pos = self.alpha_pos
        cdef double[:] beta_pos = self.beta_pos
        cdef double[:] delta_pos = self.delta_pos
        cdef double[:, :] positions = self.positions
        
        for i in range(self.population_size):
            for j in range(self.dim):
                # Alpha update
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha
                
                # Beta update
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta
                
                # Delta update
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta
                
                # Update position
                positions[i, j] = (X1 + X2 + X3) / 3

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Grey Wolf Optimizer."""
        self.initialize_positions()
        self.convergence_curve = np.zeros(self.max_iter, dtype=DTYPE)
        self.alpha_history = []
        
        cdef int iteration
        for iteration in range(self.max_iter):
            self.enforce_bounds()
            self.update_hierarchy()
            self.update_positions(iteration)
            self.convergence_curve[iteration] = self.alpha_score
            self.alpha_history.append(self.alpha_pos.copy())
            print(f"Iteration {iteration + 1}: Best Score = {self.alpha_score}")
        
        return self.alpha_pos, self.alpha_score, self.convergence_curve, self.alpha_history

# Example usage with a benchmark function (F10 from Get_Functions_details.m)
def F10(x):
    """Benchmark function F10."""
    dim = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim))
    term2 = -np.exp(np.Sum(np.cos(2 * np.pi * x)) / dim)
    return term1 + term2 + 20 + np.exp(1)

if __name__ == "__main__":
    # Parameters
    dim = 30
    bounds = (-32, 32)  # Single bound for all dimensions
    population_size = 30
    max_iter = 500
    
    # Initialize and run optimizer
    optimizer = GreyWolfOptimizer(F10, dim, bounds, population_size, max_iter)
    best_pos, best_score, convergence_curve, alpha_history = optimizer.optimize()
    
    print(f"\nBest solution: {best_pos}")
    print(f"Best score: {best_score}")
