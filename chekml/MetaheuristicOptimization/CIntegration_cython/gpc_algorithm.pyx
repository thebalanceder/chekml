# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, M_PI
from cython.parallel import prange
from libc.stdlib cimport rand, RAND_MAX

# Helper function to generate uniform random numbers in C
cdef inline double uniform(double a, double b) nogil:
    return a + (b - a) * (rand() / RAND_MAX)

cdef class GizaPyramidsConstruction:
    cdef:
        int dim, population_size, max_iter
        double G, theta, mu_min, mu_max, V_min, V_max
        object pyramids  # Declare as object
        object bounds
        object best_solution
        double best_value
        list history
        object objective_function

    def __init__(self, object objective_function, int dim, list bounds, 
                 int population_size=50, int max_iter=100, double G=9.8, 
                 double mu_min=0.3, double mu_max=0.5, 
                 double V_min=0.1, double V_max=2.0):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_iter = max_iter
        self.G = G
        self.theta = 14 * M_PI / 180  # Convert 14 degrees to radians
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.V_min = V_min
        self.V_max = V_max
        self.pyramids = np.empty((population_size, dim), dtype=np.float64)
        self.best_solution = np.empty(dim, dtype=np.float64)
        self.best_value = float("inf")
        self.history = []

    cdef void initialize_pyramids(self):
        """ Generate an initial random population """
        cdef int i, j
        cdef cnp.ndarray[cnp.float64_t, ndim=2] pyramids = self.pyramids

        for i in range(self.population_size):
            for j in range(self.dim):
                pyramids[i, j] = np.random.uniform(self.bounds[j, 0], self.bounds[j, 1])

    cdef cnp.ndarray evaluate_pyramids(self):
        """ Compute fitness values """
        cdef int i
        cdef cnp.ndarray[cnp.float64_t, ndim=1] fitness = np.empty(self.population_size, dtype=np.float64)
        
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.pyramids[i])
        return fitness

    cdef void compute_movement(self, double velocity, double friction, double* d, double* x) nogil:
        """ Compute stone and worker movement """
        d[0] = (velocity ** 2) / (2 * self.G * (sin(self.theta) + friction * cos(self.theta)))
        x[0] = (velocity ** 2) / (2 * self.G * sin(self.theta))

    cdef void update_pyramids(self):
        """ Perform physics-based movement updates """
        cdef int i, j, best_index
        cdef cnp.ndarray[cnp.float64_t, ndim=1] fitness = self.evaluate_pyramids()
        cdef double velocity, friction, d, x, epsilon

        best_index = np.argmin(fitness)
        self.best_solution[:] = self.pyramids[best_index]

        for i in prange(self.population_size, nogil=True):
            if i != best_index:
                for j in range(self.dim):
                    velocity = uniform(self.V_min, self.V_max)
                    friction = uniform(self.mu_min, self.mu_max)
                    d = 0.0
                    x = 0.0
                    self.compute_movement(velocity, friction, &d, &x)
                    epsilon = uniform(-0.5 * (self.V_max - self.V_min), 0.5 * (self.V_max - self.V_min))
                    
                    with gil:
                        self.pyramids[i, j] = (self.pyramids[i, j] + d) * (x * epsilon)
                        
                        # Ensure bounds
                        if self.pyramids[i, j] < self.bounds[j, 0]:
                            self.pyramids[i, j] = self.bounds[j, 0]
                        elif self.pyramids[i, j] > self.bounds[j, 1]:
                            self.pyramids[i, j] = self.bounds[j, 1]

    def optimize(self):
        """ Run the GPC algorithm """
        cdef int generation, min_idx
        cdef cnp.ndarray[cnp.float64_t, ndim=1] fitness
        self.initialize_pyramids()

        for generation in range(self.max_iter):
            self.update_pyramids()
            fitness = self.evaluate_pyramids()

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_value = fitness[min_idx]
                self.best_solution[:] = self.pyramids[min_idx]
            
            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")
        
        return self.best_solution, self.best_value, self.history

