# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport abs

# Define NumPy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class CulturalAlgorithmOptimizer:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        double p_accept
        int n_accept
        double alpha
        double beta
        np.ndarray population
        np.ndarray costs
        np.ndarray best_solution
        double best_cost
        list history
        dict culture

    def __init__(self, objective_function, int dim, np.ndarray[DTYPE_t, ndim=2] bounds, 
                 int population_size=50, int max_iter=1000, double p_accept=0.35, 
                 double alpha=0.3, double beta=0.5):
        """
        Initialize the Cultural Algorithm optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Array of shape (dim, 2) with (lower, upper) bounds for each dimension.
        - population_size: Number of solutions in the population.
        - max_iter: Maximum number of iterations.
        - p_accept: Acceptance ratio for selecting individuals to update culture.
        - alpha: Scaling factor for normative influence.
        - beta: Scaling factor for normative range adjustment (unused in method 3).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = bounds
        self.population_size = population_size
        self.max_iter = max_iter
        self.p_accept = p_accept
        self.n_accept = int(round(p_accept * population_size))  # Number of accepted individuals
        self.alpha = alpha
        self.beta = beta

        self.population = None
        self.costs = None
        self.best_solution = None
        self.best_cost = np.inf
        self.history = []

        # Initialize culture
        self.culture = {
            "Situational": {"Position": None, "Cost": np.inf},
            "Normative": {
                "Min": np.full(dim, np.inf, dtype=DTYPE),
                "Max": np.full(dim, -np.inf, dtype=DTYPE),
                "L": np.full(dim, np.inf, dtype=DTYPE),
                "U": np.full(dim, np.inf, dtype=DTYPE),
                "Size": np.zeros(dim, dtype=DTYPE)
            }
        }

    cpdef void initialize_population(self):
        """Generate initial population randomly within bounds."""
        cdef np.ndarray[DTYPE_t, ndim=2] population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim))
        self.population = population
        self.costs = self.evaluate_population()
        
        # Sort population by cost
        cdef np.ndarray[np.intp_t, ndim=1] sort_indices = np.argsort(self.costs)
        self.population = self.population[sort_indices]
        self.costs = self.costs[sort_indices]
        
        # Initialize culture with top n_accept individuals
        cdef np.ndarray[DTYPE_t, ndim=2] selected_pop = self.population[:self.n_accept]
        cdef np.ndarray[DTYPE_t, ndim=1] selected_costs = self.costs[:self.n_accept]
        self.adjust_culture(selected_pop, selected_costs)
        
        # Set initial best solution
        self.best_solution = self.culture["Situational"]["Position"].copy()
        self.best_cost = self.culture["Situational"]["Cost"]

    cpdef np.ndarray[DTYPE_t, ndim=1] evaluate_population(self):
        """Compute fitness values for the population."""
        cdef np.ndarray[DTYPE_t, ndim=1] costs = np.empty(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            costs[i] = self.objective_function(self.population[i])
        return costs

    cpdef void adjust_culture(self, np.ndarray[DTYPE_t, ndim=2] selected_pop, 
                             np.ndarray[DTYPE_t, ndim=1] selected_costs):
        """Update situational and normative components of the culture."""
        cdef int n = selected_pop.shape[0]
        cdef int i, j
        cdef np.ndarray[DTYPE_t, ndim=1] situational_pos = self.culture["Situational"]["Position"]
        cdef double situational_cost = self.culture["Situational"]["Cost"]
        cdef np.ndarray[DTYPE_t, ndim=1] norm_min = self.culture["Normative"]["Min"]
        cdef np.ndarray[DTYPE_t, ndim=1] norm_max = self.culture["Normative"]["Max"]
        cdef np.ndarray[DTYPE_t, ndim=1] norm_l = self.culture["Normative"]["L"]
        cdef np.ndarray[DTYPE_t, ndim=1] norm_u = self.culture["Normative"]["U"]
        cdef np.ndarray[DTYPE_t, ndim=1] norm_size = self.culture["Normative"]["Size"]

        for i in range(n):
            # Update Situational component
            if selected_costs[i] < situational_cost:
                if situational_pos is None:
                    situational_pos = selected_pop[i].copy()
                else:
                    situational_pos[:] = selected_pop[i]
                situational_cost = selected_costs[i]
            
            # Update Normative component
            for j in range(self.dim):
                if (selected_pop[i, j] < norm_min[j] or selected_costs[i] < norm_l[j]):
                    norm_min[j] = selected_pop[i, j]
                    norm_l[j] = selected_costs[i]
                if (selected_pop[i, j] > norm_max[j] or selected_costs[i] < norm_u[j]):
                    norm_max[j] = selected_pop[i, j]
                    norm_u[j] = selected_costs[i]
        
        # Update Normative Size
        for j in range(self.dim):
            norm_size[j] = norm_max[j] - norm_min[j]
        
        self.culture["Situational"]["Position"] = situational_pos
        self.culture["Situational"]["Cost"] = situational_cost
        self.culture["Normative"]["Min"] = norm_min
        self.culture["Normative"]["Max"] = norm_max
        self.culture["Normative"]["L"] = norm_l
        self.culture["Normative"]["U"] = norm_u
        self.culture["Normative"]["Size"] = norm_size

    cpdef void influence_culture(self):
        """Apply cultural influence to update population positions (Method 3)."""
        cdef np.ndarray[DTYPE_t, ndim=2] new_population = self.population.copy()
        cdef np.ndarray[DTYPE_t, ndim=1] situational_pos = self.culture["Situational"]["Position"]
        cdef np.ndarray[DTYPE_t, ndim=1] norm_size = self.culture["Normative"]["Size"]
        cdef int i, j
        cdef double sigma, dx
        cdef np.ndarray[DTYPE_t, ndim=2] randn = np.random.randn(self.population_size, self.dim)
        
        for i in range(self.population_size):
            for j in range(self.dim):
                # Compute standard deviation
                sigma = self.alpha * norm_size[j]
                dx = sigma * randn[i, j]
                
                # Adjust direction based on Situational component
                if new_population[i, j] < situational_pos[j]:
                    dx = abs(dx)
                elif new_population[i, j] > situational_pos[j]:
                    dx = -abs(dx)
                
                # Update position
                new_population[i, j] += dx
        
        # Clip to bounds
        for i in range(self.population_size):
            for j in range(self.dim):
                if new_population[i, j] < self.bounds[j, 0]:
                    new_population[i, j] = self.bounds[j, 0]
                elif new_population[i, j] > self.bounds[j, 1]:
                    new_population[i, j] = self.bounds[j, 1]
        
        self.population = new_population

    cpdef tuple optimize(self):
        """Run the Cultural Algorithm optimization."""
        self.initialize_population()
        
        cdef int iteration
        cdef np.ndarray[np.intp_t, ndim=1] sort_indices
        cdef np.ndarray[DTYPE_t, ndim=2] selected_pop
        cdef np.ndarray[DTYPE_t, ndim=1] selected_costs
        
        for iteration in range(self.max_iter):
            # Apply cultural influence
            self.influence_culture()
            
            # Evaluate new costs
            self.costs = self.evaluate_population()
            
            # Sort population by cost
            sort_indices = np.argsort(self.costs)
            self.population = self.population[sort_indices]
            self.costs = self.costs[sort_indices]
            
            # Adjust culture with top n_accept individuals
            selected_pop = self.population[:self.n_accept]
            selected_costs = self.costs[:self.n_accept]
            self.adjust_culture(selected_pop, selected_costs)
            
            # Update best solution
            if self.culture["Situational"]["Cost"] < self.best_cost:
                self.best_solution = self.culture["Situational"]["Position"].copy()
                self.best_cost = self.culture["Situational"]["Cost"]
            
            # Store history
            self.history.append((iteration, self.best_solution.copy(), self.best_cost))
            print(f"Iteration {iteration + 1}: Best Cost = {self.best_cost}")
        
        return self.best_solution, self.best_cost, self.history
