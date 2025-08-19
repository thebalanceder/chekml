# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp

# Define NumPy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class SalpSwarmAlgorithm:
    """
    Salp Swarm Algorithm (SSA) for optimization problems.
    
    Parameters:
    - objective_function: Function to optimize.
    - dim: Number of dimensions (variables).
    - bounds: Tuple of (lower, upper) bounds for each dimension.
    - population_size: Number of salp search agents (default: 30).
    - max_iter: Maximum number of iterations (default: 1000).
    """
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        np.ndarray salp_positions
        np.ndarray food_position
        double food_fitness
        list convergence_curve

    def __init__(self, objective_function, int dim, bounds, int population_size=30, int max_iter=1000):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.salp_positions = None  # Population of salps
        self.food_position = None   # Best solution (food source)
        self.food_fitness = np.inf  # Best fitness value
        self.convergence_curve = []  # History of best fitness per iteration

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void initialize_salps(self):
        """ Initialize the first population of salps """
        cdef np.ndarray[DTYPE_t, ndim=2] lb = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=2] ub = self.bounds[:, 1]
        cdef np.ndarray[DTYPE_t, ndim=2] positions
        cdef int i

        if lb.shape[0] == 1:  # Single bound for all dimensions
            positions = np.random.rand(self.population_size, self.dim) * (ub[0, 0] - lb[0, 0]) + lb[0, 0]
        else:  # Different bounds for each dimension
            positions = np.zeros((self.population_size, self.dim), dtype=DTYPE)
            for i in range(self.dim):
                positions[:, i] = np.random.rand(self.population_size) * (ub[i, 0] - lb[i, 0]) + lb[i, 0]
        
        self.salp_positions = positions

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[DTYPE_t, ndim=1] evaluate_salps(self):
        """ Compute fitness values for the salp positions """
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.salp_positions[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[DTYPE_t, ndim=1] update_leader_salp(self, int index, double c1):
        """ Update leader salp position based on food source (Eq. 3.1) """
        cdef np.ndarray[DTYPE_t, ndim=2] lb = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=2] ub = self.bounds[:, 1]
        cdef np.ndarray[DTYPE_t, ndim=1] new_position = np.empty(self.dim, dtype=DTYPE)
        cdef int j
        cdef double c2, c3

        for j in range(self.dim):
            c2 = np.random.rand()
            c3 = np.random.rand()
            if c3 < 0.5:
                new_position[j] = self.food_position[j] + c1 * ((ub[j, 0] - lb[j, 0]) * c2 + lb[j, 0])
            else:
                new_position[j] = self.food_position[j] - c1 * ((ub[j, 0] - lb[j, 0]) * c2 + lb[j, 0])
        
        # Clip to bounds
        for j in range(self.dim):
            if new_position[j] < lb[j, 0]:
                new_position[j] = lb[j, 0]
            elif new_position[j] > ub[j, 0]:
                new_position[j] = ub[j, 0]
        
        return new_position

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[DTYPE_t, ndim=1] update_follower_salp(self, int index):
        """ Update follower salp position (Eq. 3.4) """
        cdef np.ndarray[DTYPE_t, ndim=1] point1 = self.salp_positions[index - 1]
        cdef np.ndarray[DTYPE_t, ndim=1] point2 = self.salp_positions[index]
        return (point1 + point2) / 2.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self):
        """ Run the Salp Swarm Algorithm """
        # Initialize salp positions
        self.initialize_salps()
        
        # Evaluate initial fitness
        cdef np.ndarray[DTYPE_t, ndim=1] salp_fitness = self.evaluate_salps()
        
        # Initialize food source (best solution)
        cdef int best_idx = np.argmin(salp_fitness)
        self.food_position = self.salp_positions[best_idx].copy()
        self.food_fitness = salp_fitness[best_idx]
        
        # Sort salps by fitness for initial leader-follower structure
        cdef np.ndarray[long, ndim=1] sorted_indices = np.argsort(salp_fitness)
        self.salp_positions = self.salp_positions[sorted_indices]
        
        # Main loop
        cdef int l, i, j
        cdef double c1
        cdef np.ndarray[DTYPE_t, ndim=2] lb = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=2] ub = self.bounds[:, 1]
        cdef np.ndarray[DTYPE_t, ndim=1] position
        cdef bint above_ub, below_lb

        for l in range(1, self.max_iter + 1):
            # Update c1 coefficient (Eq. 3.2)
            c1 = 2 * exp(-((4 * l / self.max_iter) ** 2))
            
            # Update salp positions
            for i in range(self.population_size):
                if i < self.population_size / 2:  # Leader salps
                    self.salp_positions[i] = self.update_leader_salp(i, c1)
                else:  # Follower salps
                    self.salp_positions[i] = self.update_follower_salp(i)
            
            # Boundary checking
            for i in range(self.population_size):
                position = self.salp_positions[i]
                for j in range(self.dim):
                    above_ub = position[j] > ub[j, 0]
                    below_lb = position[j] < lb[j, 0]
                    if above_ub:
                        position[j] = ub[j, 0]
                    elif below_lb:
                        position[j] = lb[j, 0]
            
            # Evaluate fitness and update food source
            salp_fitness = self.evaluate_salps()
            for i in range(self.population_size):
                if salp_fitness[i] < self.food_fitness:
                    self.food_position = self.salp_positions[i].copy()
                    self.food_fitness = salp_fitness[i]
            
            # Store convergence data
            self.convergence_curve.append(self.food_fitness)
            print(f"Iteration {l}: Best Fitness = {self.food_fitness}")
        
        return self.food_position, self.food_fitness, self.convergence_curve

# Example usage
if __name__ == "__main__":
    # Example objective function (Sphere function)
    def sphere_function(x):
        return np.sum(x ** 2)
    
    # Parameters
    dim = 10
    bounds = [(-100, 100)] * dim  # Bounds for each dimension
    population_size = 30
    max_iter = 1000
    
    # Initialize and run SSA
    ssa = SalpSwarmAlgorithm(sphere_function, dim, bounds, population_size, max_iter)
    best_position, best_fitness, convergence = ssa.optimize()
    
    print(f"Best Solution: {best_position}")
    print(f"Best Fitness: {best_fitness}")
