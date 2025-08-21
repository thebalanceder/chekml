import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt
from libc.stdlib cimport rand, RAND_MAX
from cpython.array cimport array

# Define numpy types
ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double random_uniform(double low, double high):
    """Generate a random double between low and high."""
    return low + (high - low) * (<double>rand() / RAND_MAX)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FireHawkOptimizer:
    cdef public:
        object objective_function
        int dim
        cnp.ndarray bounds
        int population_size
        int max_fes
        int num_firehawks
        cnp.ndarray population
        cnp.ndarray best_solution
        double best_value
        list history
        int fes
        int iteration

    def __init__(self, objective_function, int dim, bounds, int population_size=25, int max_fes=100):
        """
        Initialize the Fire Hawk Optimizer (FHO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of initial candidates (Fire Hawks and Prey).
        - max_fes: Maximum number of function evaluations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_fes = max_fes
        self.num_firehawks = np.random.randint(1, int(np.ceil(population_size / 5)) + 1)
        self.population = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []
        self.fes = 0
        self.iteration = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void initialize_population(self):
        """Generate initial population randomly."""
        cdef cnp.ndarray[DTYPE_t, ndim=2] pop = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim)
        )
        cdef cnp.ndarray[DTYPE_t, ndim=1] costs = self.evaluate_population(pop)
        cdef cnp.ndarray[cnp.intp_t, ndim=1] sort_order = np.argsort(costs)
        self.population = pop[sort_order]
        self.best_solution = self.population[0].copy()
        self.best_value = costs[sort_order[0]]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[DTYPE_t, ndim=1] evaluate_population(self, cnp.ndarray[DTYPE_t, ndim=2] pop):
        """Compute fitness values for the population."""
        cdef int i
        cdef cnp.ndarray[DTYPE_t, ndim=1] costs = np.empty(pop.shape[0], dtype=np.float64)
        for i in range(pop.shape[0]):
            costs[i] = self.objective_function(pop[i])
        self.fes += pop.shape[0]
        return costs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double distance(self, cnp.ndarray[DTYPE_t, ndim=1] a, cnp.ndarray[DTYPE_t, ndim=1] b):
        """Calculate Euclidean distance between two points."""
        cdef double dist = 0.0
        cdef int i
        for i in range(self.dim):
            dist += (a[i] - b[i]) ** 2
        return sqrt(dist)

    @cython.boundscheck(True)  # Temporarily enable bounds checking
    @cython.wraparound(True)
    cdef list assign_prey_to_firehawks(self, cnp.ndarray[DTYPE_t, ndim=2] firehawks, cnp.ndarray[DTYPE_t, ndim=2] prey):
        """Assign prey to Fire Hawks based on distance."""
        cdef list prey_groups = []
        cdef list remaining_indices = list(range(prey.shape[0]))
        cdef int i, j, num_prey
        cdef cnp.ndarray[DTYPE_t, ndim=1] distances
        cdef cnp.ndarray[cnp.intp_t, ndim=1] sort_indices
        for i in range(firehawks.shape[0]):
            if not remaining_indices:
                break
            distances = np.empty(len(remaining_indices), dtype=np.float64)
            for j in range(len(remaining_indices)):
                distances[j] = self.distance(firehawks[i], prey[remaining_indices[j]])
            sort_indices = np.argsort(distances)
            num_prey = np.random.randint(1, len(remaining_indices) + 1)
            selected_indices = [remaining_indices[int(sort_indices[k])] for k in range(min(num_prey, len(sort_indices)))]
            prey_groups.append(prey[selected_indices])
            # Remove selected indices
            remaining_indices = [idx for idx in remaining_indices if idx not in selected_indices]
        if remaining_indices:
            if prey_groups:
                prey_groups[-1] = np.vstack((prey_groups[-1], prey[remaining_indices]))
            else:
                prey_groups.append(prey[remaining_indices])
        return prey_groups

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[DTYPE_t, ndim=1] update_firehawk_position(self, cnp.ndarray[DTYPE_t, ndim=1] fh, cnp.ndarray[DTYPE_t, ndim=1] other_fh):
        """Update Fire Hawk position."""
        cdef double ir1 = random_uniform(0, 1)
        cdef double ir2 = random_uniform(0, 1)
        cdef cnp.ndarray[DTYPE_t, ndim=1] new_pos = np.empty(self.dim, dtype=np.float64)
        cdef int i
        for i in range(self.dim):
            new_pos[i] = fh[i] + (ir1 * self.best_solution[i] - ir2 * other_fh[i])
            new_pos[i] = max(min(new_pos[i], self.bounds[i, 1]), self.bounds[i, 0])
        return new_pos

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef tuple update_prey_position(self, cnp.ndarray[DTYPE_t, ndim=1] prey, cnp.ndarray[DTYPE_t, ndim=1] firehawk,
                                   cnp.ndarray[DTYPE_t, ndim=1] safe_point, cnp.ndarray[DTYPE_t, ndim=1] global_safe_point):
        """Update Prey position with two strategies."""
        cdef double ir1_1 = random_uniform(0, 1)
        cdef double ir1_2 = random_uniform(0, 1)
        cdef cnp.ndarray[DTYPE_t, ndim=1] pos1 = np.empty(self.dim, dtype=np.float64)
        cdef double ir2_1 = random_uniform(0, 1)
        cdef double ir2_2 = random_uniform(0, 1)
        cdef cnp.ndarray[DTYPE_t, ndim=1] pos2 = np.empty(self.dim, dtype=np.float64)
        cdef int i
        for i in range(self.dim):
            pos1[i] = prey[i] + (ir1_1 * firehawk[i] - ir1_2 * safe_point[i])
            pos1[i] = max(min(pos1[i], self.bounds[i, 1]), self.bounds[i, 0])
            pos2[i] = prey[i] + (ir2_1 * firehawk[i] - ir2_2 * global_safe_point[i])
            pos2[i] = max(min(pos2[i], self.bounds[i, 1]), self.bounds[i, 0])
        return pos1, pos2

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self):
        """Run the Fire Hawk Optimization algorithm."""
        self.initialize_population()
        cdef cnp.ndarray[DTYPE_t, ndim=1] global_safe_point = np.mean(self.population, axis=0)
        cdef list new_population_list
        cdef cnp.ndarray[DTYPE_t, ndim=2] firehawks, prey, new_population
        cdef cnp.ndarray[DTYPE_t, ndim=1] costs, new_fh, pos1, pos2
        cdef cnp.ndarray[cnp.intp_t, ndim=1] sort_order
        cdef int i, j
        while self.fes < self.max_fes:
            self.iteration += 1
            self.num_firehawks = np.random.randint(1, int(np.ceil(self.population_size / 5)) + 1)
            firehawks = self.population[:self.num_firehawks]
            prey = self.population[self.num_firehawks:] if self.num_firehawks < self.population_size else np.array([])

            prey_groups = self.assign_prey_to_firehawks(firehawks, prey) if prey.shape[0] > 0 else []
            new_population_list = []
            for i in range(firehawks.shape[0]):
                other_fh = firehawks[np.random.randint(0, firehawks.shape[0])]
                new_fh = self.update_firehawk_position(firehawks[i], other_fh)
                new_population_list.append(new_fh)

                if i < len(prey_groups) and prey_groups[i].shape[0] > 0:
                    local_safe_point = np.mean(prey_groups[i], axis=0)
                    for j in range(prey_groups[i].shape[0]):
                        pos1, pos2 = self.update_prey_position(prey_groups[i][j], firehawks[i], local_safe_point, global_safe_point)
                        new_population_list.extend([pos1, pos2])

            if new_population_list:
                new_population = np.array(new_population_list, dtype=np.float64)
                costs = self.evaluate_population(new_population)
                sort_order = np.argsort(costs)
                new_population = new_population[sort_order]
                costs = costs[sort_order]
                self.population = new_population[:self.population_size] if new_population.shape[0] > self.population_size else new_population
                if costs[0] < self.best_value:
                    self.best_solution = self.population[0].copy()
                    self.best_value = costs[0]

            global_safe_point = np.mean(self.population, axis=0)
            self.history.append((self.iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {self.iteration}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

# Example usage
if __name__ == "__main__":
    def sphere(x):
        return np.sum(x ** 2)

    dim = 10
    bounds = [(-10, 10)] * dim
    optimizer = FireHawkOptimizer(sphere, dim, bounds)
    best_solution, best_value, history = optimizer.optimize()
    print(f"Best Solution: {best_solution}")
    print(f"Best Value: {best_value}")
