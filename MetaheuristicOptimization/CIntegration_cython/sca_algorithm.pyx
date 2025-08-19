import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sin, cos, abs as c_abs, M_PI

# Define numpy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class SineCosineAlgorithm:
    """
    Cythonized Sine Cosine Algorithm (SCA) for optimization problems.
    
    Parameters:
    - objective_function: Function to optimize.
    - dim: Number of dimensions (variables).
    - bounds: Tuple of (lower, upper) bounds for each dimension or single values if same for all.
    - population_size: Number of search agents.
    - max_iter: Maximum number of iterations.
    - a: Controls the linear decrease of r1 (default=2).
    
    Reference:
    S. Mirjalili, SCA: A Sine Cosine Algorithm for solving optimization problems
    Knowledge-Based Systems, DOI: http://dx.doi.org/10.1016/j.knosys.2015.12.022
    """
    
    cdef object objective_function
    cdef int dim, population_size
    cdef public int max_iter
    cdef double a
    cdef public np.ndarray bounds, lb, ub, solutions, best_solution
    cdef public double best_fitness
    cdef list convergence_curve

    def __init__(self, objective_function, int dim, bounds, int population_size=30, int max_iter=1000, double a=2):
        self.objective_function = objective_function
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.a = a
        
        # Convert bounds to numpy array
        self.bounds = np.array(bounds, dtype=DTYPE)
        
        # Handle bounds: single values or per dimension
        if self.bounds.ndim == 1:
            self.lb = np.full(dim, self.bounds[0], dtype=DTYPE)
            self.ub = np.full(dim, self.bounds[1], dtype=DTYPE)
        else:
            self.lb = self.bounds[:, 0]
            self.ub = self.bounds[:, 1]
        
        self.solutions = None
        self.best_solution = None
        self.best_fitness = np.inf
        self.convergence_curve = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void initialize_solutions(self):
        """Generate initial population of random solutions."""
        cdef int i
        if self.bounds.ndim == 1:
            self.solutions = np.random.uniform(self.lb[0], self.ub[0], 
                                            (self.population_size, self.dim)).astype(DTYPE)
        else:
            self.solutions = np.zeros((self.population_size, self.dim), dtype=DTYPE)
            for i in range(self.dim):
                self.solutions[:, i] = np.random.uniform(self.lb[i], self.ub[i], 
                                                      self.population_size)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray evaluate_solutions(self):
        """Compute fitness values for all solutions."""
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.solutions[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_position(self, int t):
        """Update positions of solutions using sine-cosine equations."""
        cdef double r1, r2, r3, r4
        cdef int i, j
        # Eq. (3.4): Linearly decreasing r1
        r1 = self.a - t * (self.a / self.max_iter)
        
        for i in range(self.population_size):
            for j in range(self.dim):
                # Update r2, r3, r4 for Eq. (3.3)
                r2 = 2 * M_PI * np.random.rand()
                r3 = 2 * np.random.rand()
                r4 = np.random.rand()
                
                # Eq. (3.3): Update position
                if r4 < 0.5:
                    # Eq. (3.1): Sine update
                    self.solutions[i, j] = (self.solutions[i, j] + 
                                          r1 * sin(r2) * 
                                          c_abs(r3 * self.best_solution[j] - self.solutions[i, j]))
                else:
                    # Eq. (3.2): Cosine update
                    self.solutions[i, j] = (self.solutions[i, j] + 
                                          r1 * cos(r2) * 
                                          c_abs(r3 * self.best_solution[j] - self.solutions[i, j]))
            
            # Ensure solutions stay within bounds
            for j in range(self.dim):
                if self.solutions[i, j] < self.lb[j]:
                    self.solutions[i, j] = self.lb[j]
                elif self.solutions[i, j] > self.ub[j]:
                    self.solutions[i, j] = self.ub[j]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self):
        """Run the Sine Cosine Algorithm."""
        print("SCA is optimizing your problem...")
        
        # Initialize solutions
        self.initialize_solutions()
        
        # Evaluate initial solutions and find the best
        cdef np.ndarray[DTYPE_t, ndim=1] fitness_values = self.evaluate_solutions()
        cdef int min_idx = np.argmin(fitness_values)
        self.best_solution = self.solutions[min_idx].copy()
        self.best_fitness = fitness_values[min_idx]
        self.convergence_curve.append(self.best_fitness)
        
        # Main loop
        cdef int t
        for t in range(1, self.max_iter):
            # Update positions
            self.update_position(t)
            
            # Evaluate new solutions
            fitness_values = self.evaluate_solutions()
            
            # Update best solution if a better one is found
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < self.best_fitness:
                self.best_solution = self.solutions[min_idx].copy()
                self.best_fitness = fitness_values[min_idx]
            
            # Store convergence data
            self.convergence_curve.append(self.best_fitness)
            
            # Display progress every 50 iterations
            if (t + 1) % 50 == 0:
                print(f"At iteration {t + 1}, the optimum is {self.best_fitness}")
        
        print(f"The best solution obtained by SCA is: {self.best_solution}")
        print(f"The best optimal value of the objective function found by SCA is: {self.best_fitness}")
        
        return self.best_solution, self.best_fitness, self.convergence_curve

# Example usage (for testing, will be compiled separately)
if __name__ == "__main__":
    def sphere_function(x):
        return np.sum(x ** 2)
    
    dim = 30
    bounds = [(-100, 100)] * dim
    population_size = 30
    max_iter = 1000
    
    sca = SineCosineAlgorithm(sphere_function, dim, bounds, population_size, max_iter)
    best_solution, best_fitness, convergence_curve = sca.optimize()
