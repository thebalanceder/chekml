# cython: language_level=3
# distutils: extra_compile_args = -O3 -ffast-math

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, cos, fabs, pow
from libc.stdlib cimport rand, RAND_MAX
import cython

# Type definitions
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class DujiangyanIrrigationOptimizer:
    cdef object objective_function
    cdef int dim
    cdef np.ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef DTYPE_t diversion_factor
    cdef DTYPE_t flow_adjustment
    cdef DTYPE_t water_density
    cdef DTYPE_t fluid_distribution
    cdef DTYPE_t centrifugal_resistance
    cdef DTYPE_t bottleneck_ratio
    cdef DTYPE_t elimination_ratio
    
    cdef np.ndarray water_flows
    cdef np.ndarray best_solution
    cdef DTYPE_t best_value
    cdef list history
    
    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100, 
                 DTYPE_t diversion_factor=0.3, DTYPE_t flow_adjustment=0.2, DTYPE_t water_density=1.35, 
                 DTYPE_t fluid_distribution=0.46, DTYPE_t centrifugal_resistance=1.2, 
                 DTYPE_t bottleneck_ratio=0.68, DTYPE_t elimination_ratio=0.23):
        
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.diversion_factor = diversion_factor
        self.flow_adjustment = flow_adjustment
        self.water_density = water_density
        self.fluid_distribution = fluid_distribution
        self.centrifugal_resistance = centrifugal_resistance
        self.bottleneck_ratio = bottleneck_ratio
        self.elimination_ratio = elimination_ratio
        
        self.water_flows = None
        self.best_solution = None
        self.best_value = np.inf
        self.history = []
    
    cpdef initialize_water_flows(self):
        """ Generate initial water flow paths randomly """
        cdef np.ndarray[DTYPE_t, ndim=1] lower_bounds = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] upper_bounds = self.bounds[:, 1]
        
        self.water_flows = np.random.uniform(lower_bounds, upper_bounds, 
                                           (self.population_size, self.dim))
    
    cpdef evaluate_water_flows(self):
        """ Compute fitness values for the water flow paths """
        cdef int i
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=DTYPE)
        
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.water_flows[i])
        
        return fitness
    
    cpdef diversion_phase(self, int index):
        """ Simulate the Fish Mouth Dividing Project (global search) """
        cdef DTYPE_t r1 = rand() / <DTYPE_t>RAND_MAX
        cdef DTYPE_t r2 = rand() / <DTYPE_t>RAND_MAX
        cdef DTYPE_t HRO = 1.2, HRI = 7.2, HGO = 1.3, HGI = 0.82
        cdef DTYPE_t CFR = 9.435 * np.random.gamma(0.85, 2.5)  # Comprehensive riverbed roughness
        cdef DTYPE_t Vi1
        cdef np.ndarray[DTYPE_t, ndim=1] new_solution
        
        if r1 < 0.23:
            Vi1 = pow(fabs(HRO), (2.0/3.0)) * pow(fabs(HGO), (1.0/2.0)) / CFR * r1
        else:
            Vi1 = pow(fabs(HRI), (2.0/3.0)) * pow(fabs(HGI), (1.0/2.0)) / CFR * r2
        
        cdef np.ndarray[DTYPE_t, ndim=1] current_best = (self.best_solution if self.best_solution is not None 
                                                         else self.water_flows[index])
        
        new_solution = current_best + (current_best - self.water_flows[index]) * np.random.rand(self.dim) * Vi1
        
        # Clip to bounds
        cdef int i
        for i in range(self.dim):
            if new_solution[i] < self.bounds[i, 0]:
                new_solution[i] = self.bounds[i, 0]
            elif new_solution[i] > self.bounds[i, 1]:
                new_solution[i] = self.bounds[i, 1]
        
        return new_solution
    
    cpdef spiral_motion_update(self, np.ndarray[DTYPE_t, ndim=2] X, 
                              np.ndarray[DTYPE_t, ndim=1] X_best, 
                              int t):
        """ 
        Simulate inner river flow with centrifugal & lateral pressure effects.
        """
        cdef int num_solutions = X.shape[0]
        cdef int num_dimensions = X.shape[1]
        cdef int T = self.max_iter
        cdef np.ndarray[DTYPE_t, ndim=2] RCF = np.empty((num_solutions, 1), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] fitness
        cdef DTYPE_t MLV
        cdef np.ndarray[DTYPE_t, ndim=2] new_X = np.empty_like(X)
        cdef int i, j
        cdef DTYPE_t sum_sq
        
        # Compute centrifugal force (RCF)
        for i in range(num_solutions):
            sum_sq = 0.0
            for j in range(num_dimensions):
                sum_sq += (X_best[j] - X[i,j]) ** 2
            RCF[i,0] = self.water_density * cos(90 * (t / T)) * sqrt(sum_sq)
        
        # Compute mean longitudinal velocity (MLV)
        fitness = self.evaluate_water_flows()
        MLV = np.mean(fitness)
        
        # Compute lateral pressure (LP)
        cdef DTYPE_t LP = (self.water_density * self.fluid_distribution * MLV ** 2) / self.centrifugal_resistance
        
        # Apply spiral motion condition
        cdef np.ndarray[DTYPE_t, ndim=1] LB = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] UB = self.bounds[:, 1]
        
        for i in range(num_solutions):
            if RCF[i,0] < LP:
                for j in range(num_dimensions):
                    new_X[i,j] = X[i,j]
            else:
                for j in range(num_dimensions):
                    new_X[i,j] = (UB[j] - LB[j]) * (rand() / <DTYPE_t>RAND_MAX) + LB[j]
        
        return new_X
    
    cpdef local_development_phase(self, int index):
        """ Simulate Baopingkou Project (local refinement) """
        cdef DTYPE_t r3 = rand() / <DTYPE_t>RAND_MAX
        cdef DTYPE_t r4 = rand() / <DTYPE_t>RAND_MAX
        cdef DTYPE_t HRI = 7.2, HGI = 0.82
        cdef DTYPE_t CFR = 9.435 * np.random.gamma(0.85, 2.5)
        cdef DTYPE_t Vi2
        cdef np.ndarray[DTYPE_t, ndim=1] Improve2, new_solution
        cdef DTYPE_t fitness_diff
        
        if r3 < self.bottleneck_ratio:
            Vi2 = pow(fabs(HRI), (2.0/3.0)) * pow(fabs(HGI), (1.0/2.0)) / (2 * CFR) * r3
        else:
            Vi2 = pow(fabs(HRI), (2.0/3.0)) * pow(fabs(HGI), (1.0/2.0)) / (2 * CFR) * r4
        
        fitness_diff = self.best_value - self.evaluate_water_flows()[index]
        Improve2 = np.sign(fitness_diff) * (self.best_solution - self.water_flows[index]) * np.random.rand(self.dim)
        
        new_solution = self.best_solution + (self.best_solution - self.water_flows[index]) * \
                      np.random.rand(self.dim) * Vi2 + Improve2
        
        # Clip to bounds
        cdef int i
        for i in range(self.dim):
            if new_solution[i] < self.bounds[i, 0]:
                new_solution[i] = self.bounds[i, 0]
            elif new_solution[i] > self.bounds[i, 1]:
                new_solution[i] = self.bounds[i, 1]
        
        return new_solution
    
    cpdef elimination_phase(self):
        """ Simulate Feishayan Sediment Discharge (worst solution replacement) """
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = self.evaluate_water_flows()
        cdef int num_to_replace = int(self.elimination_ratio * self.population_size)
        cdef np.ndarray[long, ndim=1] worst_indices = np.argsort(fitness)[-num_to_replace:]
        cdef int i, j
        
        for i in worst_indices:
            for j in range(self.dim):
                self.water_flows[i,j] = self.bounds[j,0] + (self.bounds[j,1] - self.bounds[j,0]) * (rand() / <DTYPE_t>RAND_MAX)
    
    cpdef optimize(self):
        """ Run the Dujiangyan Irrigation System Optimization """
        self.initialize_water_flows()
        
        cdef int generation, i
        cdef np.ndarray[DTYPE_t, ndim=1] fitness
        cdef int min_idx
        
        for generation in range(self.max_iter):
            fitness = self.evaluate_water_flows()
            min_idx = np.argmin(fitness)
            
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.water_flows[min_idx].copy()
                self.best_value = fitness[min_idx]
            
            # Global search (Fish Mouth Dividing Project)
            for i in range(self.population_size):
                self.water_flows[i] = self.diversion_phase(i)
            
            # Spiral motion update (inner river flow)
            self.water_flows = self.spiral_motion_update(self.water_flows, self.best_solution, generation)
            
            # Local development (Baopingkou Project)
            for i in range(self.population_size):
                self.water_flows[i] = self.local_development_phase(i)
            
            # Elimination (Feishayan Sediment Discharge)
            self.elimination_phase()
            
            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")
        
        return self.best_solution, self.best_value, self.history
