# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport cos, sin, sqrt, fabs
from libc.stdlib cimport rand, RAND_MAX, malloc, free
from libc.float cimport DBL_MAX

# Define types for NumPy arrays
ctypedef np.float64_t DTYPE_t

# Define a struct to represent an object
cdef struct Object:
    double* position
    double cost
    double delta

# Define a struct to represent a whirlpool
cdef struct Whirlpool:
    double* position
    double cost
    double delta
    int n_objects
    Object* objects

# Define a key function for sorting objects by cost
def sort_key(obj):
    return obj[0]

cdef class TurbulentFlowWaterOptimizer:
    cdef:
        public object objective_function
        public int dim
        public np.ndarray bounds
        public int n_whirlpools
        int n_objects_per_whirlpool
        int n_pop
        int n_objects
        public int max_iter  # Accessible from Python
        Whirlpool* whirlpools
        np.ndarray best_solution
        double best_value
        np.ndarray best_costs
        np.ndarray mean_costs

    def __init__(self, object objective_function, int dim, bounds, int n_whirlpools=3, int n_objects_per_whirlpool=30, int max_iter=100):
        """
        Initialize the Turbulent Flow of Water-based Optimization (TFWO) algorithm.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - n_whirlpools: Number of whirlpools (groups of solutions).
        - n_objects_per_whirlpool: Number of objects (particles) per whirlpool.
        - max_iter: Maximum number of iterations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)  # Bounds as [(low, high), ...]
        self.n_whirlpools = n_whirlpools
        self.n_objects_per_whirlpool = n_objects_per_whirlpool
        self.n_pop = n_whirlpools + n_whirlpools * n_objects_per_whirlpool
        self.n_objects = self.n_pop - n_whirlpools
        self.max_iter = max_iter
        self.best_value = DBL_MAX
        self.best_costs = np.zeros(max_iter, dtype=np.float64)
        self.mean_costs = np.zeros(max_iter, dtype=np.float64)
        self.whirlpools = NULL

    def __dealloc__(self):
        """Free allocated memory for whirlpools and objects"""
        cdef int i, j
        if self.whirlpools != NULL:
            for i in range(self.n_whirlpools):
                if self.whirlpools[i].objects != NULL:
                    for j in range(self.whirlpools[i].n_objects):
                        if self.whirlpools[i].objects[j].position != NULL:
                            free(self.whirlpools[i].objects[j].position)
                    free(self.whirlpools[i].objects)
                if self.whirlpools[i].position != NULL:
                    free(self.whirlpools[i].position)
            free(self.whirlpools)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void initialize_whirlpools(self):
        """Generate initial population and organize into whirlpools"""
        cdef int i, j, k, obj_idx
        cdef double cost
        cdef np.ndarray[DTYPE_t, ndim=1] position
        # Temporary list to store objects for sorting
        objects = []
        for i in range(self.n_pop):
            position = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
            cost = self.objective_function(position)
            objects.append((cost, position))

        # Sort objects by cost using a separate key function
        objects = sorted(objects, key=sort_key)

        # Allocate memory for whirlpools
        self.whirlpools = <Whirlpool*>malloc(self.n_whirlpools * sizeof(Whirlpool))
        if self.whirlpools == NULL:
            raise MemoryError("Cannot allocate memory for whirlpools")

        # Initialize whirlpools
        for i in range(self.n_whirlpools):
            self.whirlpools[i].position = <double*>malloc(self.dim * sizeof(double))
            if self.whirlpools[i].position == NULL:
                raise MemoryError("Cannot allocate memory for whirlpool position")
            for k in range(self.dim):
                self.whirlpools[i].position[k] = objects[i][1][k]
            self.whirlpools[i].cost = objects[i][0]
            self.whirlpools[i].delta = 0.0
            self.whirlpools[i].n_objects = self.n_objects_per_whirlpool
            self.whirlpools[i].objects = <Object*>malloc(self.n_objects_per_whirlpool * sizeof(Object))
            if self.whirlpools[i].objects == NULL:
                raise MemoryError("Cannot allocate memory for objects")
            for j in range(self.n_objects_per_whirlpool):
                self.whirlpools[i].objects[j].position = <double*>malloc(self.dim * sizeof(double))
                if self.whirlpools[i].objects[j].position == NULL:
                    raise MemoryError("Cannot allocate memory for object position")
                self.whirlpools[i].objects[j].cost = 0.0
                self.whirlpools[i].objects[j].delta = 0.0

        # Distribute remaining objects to whirlpools
        remaining_objects = objects[self.n_whirlpools:]
        np.random.shuffle(remaining_objects)
        obj_idx = 0
        for i in range(self.n_whirlpools):
            for j in range(self.n_objects_per_whirlpool):
                for k in range(self.dim):
                    self.whirlpools[i].objects[j].position[k] = remaining_objects[obj_idx][1][k]
                self.whirlpools[i].objects[j].cost = remaining_objects[obj_idx][0]
                self.whirlpools[i].objects[j].delta = 0.0
                obj_idx += 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void effects_of_whirlpools(self, int iter):
        """Implement pseudocodes 1-5: Update objects and whirlpool positions"""
        cdef int i, j, t, k, min_idx, max_idx
        cdef double eee, fr0, fr10, cost, fr, sum_t, sum_j
        cdef np.ndarray[DTYPE_t, ndim=1] d, d2, x, RR, J, new_position
        cdef double* temp_position

        for i in range(self.n_whirlpools):
            for j in range(self.whirlpools[i].n_objects):
                # Compute influence from other whirlpools
                if self.n_whirlpools > 1:
                    J = np.zeros(self.n_whirlpools - 1, dtype=np.float64)
                    k = 0
                    for t in range(self.n_whirlpools):
                        if t != i:
                            sum_t = 0.0
                            sum_j = 0.0
                            for dim in range(self.dim):
                                sum_t += self.whirlpools[t].position[dim]
                                sum_j += self.whirlpools[i].objects[j].position[dim]
                            J[k] = (fabs(self.whirlpools[t].cost) ** 1) * (fabs(sum_t - sum_j)) ** 0.5
                            k += 1
                    min_idx = np.argmin(J)
                    max_idx = np.argmax(J)
                    if min_idx >= i:
                        min_idx += 1
                    if max_idx >= i:
                        max_idx += 1

                    d = np.random.rand(self.dim) * (np.array([self.whirlpools[min_idx].position[m] for m in range(self.dim)]) - 
                                                   np.array([self.whirlpools[i].objects[j].position[m] for m in range(self.dim)]))
                    d2 = np.random.rand(self.dim) * (np.array([self.whirlpools[max_idx].position[m] for m in range(self.dim)]) - 
                                                    np.array([self.whirlpools[i].objects[j].position[m] for m in range(self.dim)]))
                else:
                    d = np.random.rand(self.dim) * (np.array([self.whirlpools[i].position[m] for m in range(self.dim)]) - 
                                                   np.array([self.whirlpools[i].objects[j].position[m] for m in range(self.dim)]))
                    d2 = np.zeros(self.dim, dtype=np.float64)
                    min_idx = i

                # Update delta
                self.whirlpools[i].objects[j].delta += (<double>rand() / RAND_MAX) * (<double>rand() / RAND_MAX) * np.pi
                eee = self.whirlpools[i].objects[j].delta
                fr0 = cos(eee)
                fr10 = -sin(eee)

                # Compute new position
                x = ((fr0 * d) + (fr10 * d2)) * (1 + fabs(fr0 * fr10))
                RR = np.array([self.whirlpools[i].position[m] for m in range(self.dim)]) - x
                for dim in range(self.dim):
                    if RR[dim] < self.bounds[dim, 0]:
                        RR[dim] = self.bounds[dim, 0]
                    elif RR[dim] > self.bounds[dim, 1]:
                        RR[dim] = self.bounds[dim, 1]
                cost = self.objective_function(RR)

                # Update if better
                if cost <= self.whirlpools[i].objects[j].cost:
                    self.whirlpools[i].objects[j].cost = cost
                    for dim in range(self.dim):
                        self.whirlpools[i].objects[j].position[dim] = RR[dim]

                # Pseudocode 3: Random jump
                FE_i = (fabs(cos(eee) ** 2 * sin(eee) ** 2)) ** 2
                if <double>rand() / RAND_MAX < FE_i:
                    k = rand() % self.dim
                    self.whirlpools[i].objects[j].position[k] = self.bounds[k, 0] + (<double>rand() / RAND_MAX) * (self.bounds[k, 1] - self.bounds[k, 0])
                    temp_position = <double*>malloc(self.dim * sizeof(double))
                    for dim in range(self.dim):
                        temp_position[dim] = self.whirlpools[i].objects[j].position[dim]
                    self.whirlpools[i].objects[j].cost = self.objective_function(np.array([temp_position[m] for m in range(self.dim)]))
                    free(temp_position)

        # Pseudocode 4: Update whirlpool positions
        cdef double best_cost = DBL_MAX
        cdef int best_whirlpool_idx = 0
        for i in range(self.n_whirlpools):
            if self.whirlpools[i].cost < best_cost:
                best_cost = self.whirlpools[i].cost
                best_whirlpool_idx = i
        best_position = np.array([self.whirlpools[best_whirlpool_idx].position[m] for m in range(self.dim)])

        for i in range(self.n_whirlpools):
            J = np.zeros(self.n_whirlpools - 1, dtype=np.float64)
            k = 0
            for t in range(self.n_whirlpools):
                if t != i:
                    sum_t = 0.0
                    sum_j = 0.0
                    for dim in range(self.dim):
                        sum_t += self.whirlpools[t].position[dim]
                        sum_j += self.whirlpools[i].position[dim]
                    J[k] = self.whirlpools[t].cost * fabs(sum_t - sum_j)
                    k += 1
            min_idx = np.argmin(J)
            if min_idx >= i:
                min_idx += 1

            self.whirlpools[i].delta += (<double>rand() / RAND_MAX) * (<double>rand() / RAND_MAX) * np.pi
            d = np.array([self.whirlpools[min_idx].position[m] for m in range(self.dim)]) - np.array([self.whirlpools[i].position[m] for m in range(self.dim)])
            fr = fabs(cos(self.whirlpools[i].delta) + sin(self.whirlpools[i].delta))
            x = fr * np.random.rand(self.dim) * d

            new_position = np.array([self.whirlpools[min_idx].position[m] for m in range(self.dim)]) - x
            for dim in range(self.dim):
                if new_position[dim] < self.bounds[dim, 0]:
                    new_position[dim] = self.bounds[dim, 0]
                elif new_position[dim] > self.bounds[dim, 1]:
                    new_position[dim] = self.bounds[dim, 1]
            new_cost = self.objective_function(new_position)

            # Pseudocode 5: Selection
            if new_cost <= self.whirlpools[i].cost:
                self.whirlpools[i].cost = new_cost
                for dim in range(self.dim):
                    self.whirlpools[i].position[dim] = new_position[dim]

            if best_cost < self.whirlpools[best_whirlpool_idx].cost:
                self.whirlpools[i].cost = best_cost
                for dim in range(self.dim):
                    self.whirlpools[i].position[dim] = best_position[dim]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void pseudocode6(self):
        """Implement pseudocode 6: Update whirlpool with best object if better"""
        cdef int i, j, min_cost_idx
        cdef double min_cost
        cdef double* temp_position
        for i in range(self.n_whirlpools):
            min_cost = DBL_MAX
            min_cost_idx = 0
            for j in range(self.whirlpools[i].n_objects):
                if self.whirlpools[i].objects[j].cost < min_cost:
                    min_cost = self.whirlpools[i].objects[j].cost
                    min_cost_idx = j

            if min_cost <= self.whirlpools[i].cost:
                # Swap positions and costs
                temp_position = self.whirlpools[i].objects[min_cost_idx].position
                self.whirlpools[i].objects[min_cost_idx].position = <double*>malloc(self.dim * sizeof(double))
                for dim in range(self.dim):
                    self.whirlpools[i].objects[min_cost_idx].position[dim] = self.whirlpools[i].position[dim]
                self.whirlpools[i].position = temp_position
                self.whirlpools[i].objects[min_cost_idx].cost, self.whirlpools[i].cost = self.whirlpools[i].cost, min_cost

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the TFWO optimization algorithm"""
        cdef int iter, i
        cdef double best_cost
        self.initialize_whirlpools()
        
        for iter in range(self.max_iter):
            # Update whirlpools and objects
            self.effects_of_whirlpools(iter)
            self.pseudocode6()

            # Find best whirlpool
            best_cost = DBL_MAX
            best_idx = 0
            for i in range(self.n_whirlpools):
                if self.whirlpools[i].cost < best_cost:
                    best_cost = self.whirlpools[i].cost
                    best_idx = i
            best_position = np.array([self.whirlpools[best_idx].position[m] for m in range(self.dim)])

            # Update global best
            if best_cost < self.best_value:
                self.best_solution = best_position.copy()
                self.best_value = best_cost

            # Store iteration results
            self.best_costs[iter] = best_cost
            whirlpool_costs = np.zeros(self.n_whirlpools, dtype=np.float64)
            for i in range(self.n_whirlpools):
                whirlpool_costs[i] = self.whirlpools[i].cost
            self.mean_costs[iter] = np.mean(whirlpool_costs)

            print(f"Iter {iter + 1}: Best Cost = {best_cost}")

        return self.best_solution, self.best_value, {"best_costs": self.best_costs, "mean_costs": self.mean_costs}
