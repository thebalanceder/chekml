# ans_algorithm.pyx
import numpy as np
cimport numpy as np

cdef class AcrossNeighborhoodSearch:
    cdef public object objective_function
    cdef public int dim, num_neighborhoods, max_iterations
    cdef public double mutation_rate
    cdef public np.ndarray lower_bounds, upper_bounds
    cdef public np.ndarray populations
    cdef public np.ndarray fitness
    cdef public np.ndarray best_solution
    cdef public double best_fitness
    cdef list history

    def __init__(self, object objective_function, int dim=2, bounds=None,
                 int num_neighborhoods=5, double neighborhood_radius=0.1,
                 int max_iterations=100, double mutation_rate=0.1):
        """
        Initialize the Across Neighborhood Search (ANS) optimizer.
        
        Parameters:
        - objective_function: Function to be minimized.
        - dim: Number of decision variables.
        - bounds: List of (lower, upper) tuples for each dimension.
        - num_neighborhoods: Number of neighborhoods (agents).
        - neighborhood_radius: Not used directly in this implementation.
        - max_iterations: Maximum number of iterations.
        - mutation_rate: Step size for updates.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.num_neighborhoods = num_neighborhoods
        self.max_iterations = max_iterations
        self.mutation_rate = mutation_rate

        if bounds is None:
            bounds = [(-5.0, 5.0)] * dim

        self.lower_bounds = np.array([b[0] for b in bounds], dtype=np.float64)
        self.upper_bounds = np.array([b[1] for b in bounds], dtype=np.float64)

        self.populations = np.random.uniform(self.lower_bounds, self.upper_bounds,
                                              (self.num_neighborhoods, self.dim)).astype(np.float64)
        self.fitness = np.full(self.num_neighborhoods, np.inf, dtype=np.float64)
        self.best_solution = np.zeros(self.dim, dtype=np.float64)
        self.best_fitness = float("inf")
        self.history = []

    cpdef tuple optimize(self):
        """
        Run the optimization process.
        
        Returns:
         - best_solution: The best solution (numpy array).
         - best_fitness: The best fitness value found.
         - history: A list of tuples (iteration, best_solution) for each iteration.
        """
        cdef int iteration, i, neighbor_index, current_best_idx
        cdef np.ndarray direction

        for iteration in range(self.max_iterations):
            # Evaluate fitness for each neighborhood
            for i in range(self.num_neighborhoods):
                self.fitness[i] = self.objective_function(self.populations[i])
            
            # Update positions for each neighborhood
            for i in range(self.num_neighborhoods):
                neighbor_index = np.random.randint(0, self.num_neighborhoods - 1)
                if neighbor_index >= i:
                    neighbor_index += 1

                direction = self.populations[neighbor_index] - self.populations[i]
                self.populations[i] += self.mutation_rate * direction
                # Enforce boundary constraints
                self.populations[i] = np.clip(self.populations[i], self.lower_bounds, self.upper_bounds)

            current_best_idx = int(np.argmin(self.fitness))
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_solution = self.populations[current_best_idx].copy()

            self.history.append((iteration, self.best_solution.copy()))
            print("Iteration %d: Best Fitness = %f" % (iteration + 1, self.best_fitness))

        return self.best_solution, self.best_fitness, self.history

