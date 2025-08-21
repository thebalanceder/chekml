# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

# Declare NumPy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# Sphere function
def sphere(x):
    """Sphere function for optimization."""
    return np.sum(np.asarray(x) ** 2)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FrogLeapingOptimizer:
    cdef public:
        object objective_function
        int dim, max_iter, memeplex_size, num_memeplexes, population_size
        int num_parents, num_offsprings, max_fla_iter
        double step_size
        double[:, :] bounds
        double[:, :] population
        double[:] best_solution
        double best_value
        list history
        int[:, :] memeplex_indices

    def __init__(self, objective_function=None, int dim=10, bounds=(-10, 10), 
                 int max_iter=1000, int memeplex_size=10, int num_memeplexes=5, 
                 num_parents=None, int num_offsprings=3, int max_fla_iter=5, double step_size=2):
        """
        Initialize the Shuffled Frog Leaping Algorithm (SFLA) optimizer.
        """
        self.objective_function = objective_function if objective_function is not None else sphere
        self.dim = dim
        self.bounds = np.array([bounds] * dim, dtype=DTYPE)
        self.max_iter = max_iter
        self.memeplex_size = max(memeplex_size, dim + 1)
        self.num_memeplexes = num_memeplexes
        self.population_size = self.memeplex_size * num_memeplexes
        self.num_parents = num_parents if num_parents is not None else max(<int>(0.3 * self.memeplex_size), 2)
        self.num_offsprings = num_offsprings
        self.max_fla_iter = max_fla_iter
        self.step_size = step_size
        self.best_value = float("inf")
        self.history = []
        self.memeplex_indices = np.reshape(np.arange(self.population_size), (num_memeplexes, self.memeplex_size)).astype(np.int32)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_population(self):
        """Generate initial population of frogs randomly."""
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim)).astype(DTYPE)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] evaluate_population(self):
        """Compute fitness values for the population."""
        cdef double[:] fitness = np.empty(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.population[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def sort_population(self, double[:, :] population=None, double[:] fitness=None):
        """Sort population by fitness."""
        if population is None:
            population = self.population
            fitness = self.evaluate_population()
        
        # Get sort order
        cdef int[:] sort_order = np.argsort(fitness).astype(np.int32)
        cdef int n = population.shape[0]
        cdef int d = population.shape[1]
        cdef double[:, :] sorted_pop = np.empty((n, d), dtype=DTYPE)
        cdef int i, j, idx
        
        # Manually copy rows according to sort_order
        for i in range(n):
            idx = sort_order[i]
            for j in range(d):
                sorted_pop[i, j] = population[idx, j]
        
        return sorted_pop, sort_order

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bool is_in_range(self, double[:] x):
        """Check if position is within bounds."""
        cdef int i
        for i in range(self.dim):
            if x[i] < self.bounds[i, 0] or x[i] > self.bounds[i, 1]:
                return False
        return True

    def rand_sample(self, double[:] probabilities, int num_samples, bool replacement=False):
        """Random sampling with given probabilities."""
        return np.random.choice(len(probabilities), size=num_samples, replace=replacement, 
                               p=np.asarray(probabilities) / np.sum(probabilities))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:, :] run_fla(self, double[:, :] memeplex):
        """Run Frog Leaping Algorithm on a memeplex."""
        cdef int n_pop = memeplex.shape[0]
        cdef double[:] P = 2 * (n_pop + 1 - np.arange(1, n_pop + 1)) / (n_pop * (n_pop + 1))
        cdef double[:] lower_bound = memeplex[0].copy()
        cdef double[:] upper_bound = memeplex[0].copy()
        cdef int i, j, k, idx
        cdef double[:] new_solution = np.empty(self.dim, dtype=DTYPE)
        cdef double[:] step = np.empty(self.dim, dtype=DTYPE)
        cdef double new_cost
        cdef bool improvement_step2, censorship

        # Calculate memeplex range
        for i in range(1, n_pop):
            for j in range(self.dim):
                if memeplex[i, j] < lower_bound[j]:
                    lower_bound[j] = memeplex[i, j]
                if memeplex[i, j] > upper_bound[j]:
                    upper_bound[j] = memeplex[i, j]

        # FLA main loop
        for _ in range(self.max_fla_iter):
            # Select parents
            parent_indices = self.rand_sample(P, self.num_parents)
            subcomplex = memeplex[parent_indices].copy()
            sub_fitness = np.array([self.objective_function(subcomplex[i]) for i in range(self.num_parents)])

            # Generate offsprings
            for _ in range(self.num_offsprings):
                # Sort subcomplex
                sorted_indices = np.argsort(sub_fitness)
                subcomplex = subcomplex[sorted_indices]
                parent_indices = parent_indices[sorted_indices]
                sub_fitness = sub_fitness[sorted_indices]

                # Improvement Step 1: Move worst towards best in subcomplex
                for j in range(self.dim):
                    new_solution[j] = subcomplex[self.num_parents-1, j]
                    step[j] = self.step_size * np.random.rand() * (subcomplex[0, j] - subcomplex[self.num_parents-1, j])
                    new_solution[j] += step[j]
                improvement_step2 = False
                censorship = False

                if self.is_in_range(new_solution):
                    new_cost = self.objective_function(new_solution)
                    if new_cost < sub_fitness[self.num_parents-1]:
                        for j in range(self.dim):
                            subcomplex[self.num_parents-1, j] = new_solution[j]
                        sub_fitness[self.num_parents-1] = new_cost
                    else:
                        improvement_step2 = True
                else:
                    improvement_step2 = True

                # Improvement Step 2: Move worst towards global best
                if improvement_step2:
                    for j in range(self.dim):
                        new_solution[j] = subcomplex[self.num_parents-1, j]
                        step[j] = self.step_size * np.random.rand() * (self.best_solution[j] - subcomplex[self.num_parents-1, j])
                        new_solution[j] += step[j]
                    if self.is_in_range(new_solution):
                        new_cost = self.objective_function(new_solution)
                        if new_cost < sub_fitness[self.num_parents-1]:
                            for j in range(self.dim):
                                subcomplex[self.num_parents-1, j] = new_solution[j]
                            sub_fitness[self.num_parents-1] = new_cost
                        else:
                            censorship = True
                    else:
                        censorship = True

                # Censorship
                if censorship:
                    for j in range(self.dim):
                        subcomplex[self.num_parents-1, j] = np.random.uniform(lower_bound[j], upper_bound[j])
                    sub_fitness[self.num_parents-1] = self.objective_function(subcomplex[self.num_parents-1])

                # Update memeplex
                memeplex[parent_indices] = subcomplex

        return memeplex

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Shuffled Frog Leaping Algorithm."""
        self.initialize_population()
        self.population, _ = self.sort_population()
        self.best_solution = self.population[0].copy()
        self.best_value = self.objective_function(self.best_solution)

        cdef int iteration, j, k, idx
        cdef double[:, :] memeplex = np.empty((self.memeplex_size, self.dim), dtype=DTYPE)
        for iteration in range(self.max_iter):
            for j in range(self.num_memeplexes):
                # Copy memeplex rows manually
                for k in range(self.memeplex_size):
                    idx = self.memeplex_indices[j, k]
                    for d in range(self.dim):
                        memeplex[k, d] = self.population[idx, d]
                memeplex = self.run_fla(memeplex)
                # Update population
                for k in range(self.memeplex_size):
                    idx = self.memeplex_indices[j, k]
                    for d in range(self.dim):
                        self.population[idx, d] = memeplex[k, d]

            self.population, _ = self.sort_population()
            self.best_solution = self.population[0].copy()
            self.best_value = self.objective_function(self.best_solution)

            self.history.append((iteration, np.array(self.best_solution), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return np.array(self.best_solution), self.best_value, self.history
