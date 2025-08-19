import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sin, cos, sqrt, abs
from cpython cimport bool

# Declare numpy types
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class ChemicalReactionOptimizer:
    cdef object objective_function
    cdef int dim
    cdef cnp.ndarray bounds
    cdef int population_size
    cdef int max_iter
    cdef double initial_ke
    cdef double mole_coll
    cdef double buffer
    cdef double alpha
    cdef double beta
    cdef double split_ratio
    cdef double elimination_ratio
    cdef cnp.ndarray molecules
    cdef cnp.ndarray kinetic_energies
    cdef cnp.ndarray best_solution
    cdef double best_value
    cdef list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100, 
                 double initial_ke=1000.0, double mole_coll=0.5, double buffer=0.0, double alpha=10.0, 
                 double beta=0.2, double split_ratio=0.5, double elimination_ratio=0.2):
        """
        Initialize the Chemical Reaction Optimization (CRO) algorithm.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.initial_ke = initial_ke
        self.mole_coll = mole_coll
        self.buffer = buffer
        self.alpha = alpha
        self.beta = beta
        self.split_ratio = split_ratio
        self.elimination_ratio = elimination_ratio
        self.molecules = None
        self.kinetic_energies = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_molecules(self):
        """ Generate initial molecules randomly """
        self.molecules = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim)).astype(np.double)
        self.kinetic_energies = np.full(self.population_size, self.initial_ke, dtype=np.double)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_molecules(self):
        """ Compute potential energy (fitness) for the molecules """
        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness = np.empty(self.population_size, dtype=np.double)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.molecules[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def on_wall_collision(self, int index):
        """ Simulate on-wall ineffective collision (local search) """
        cdef double r = np.random.rand()
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_solution = self.molecules[index] + r * (self.bounds[:, 1] - self.bounds[:, 0]) * np.random.randn(self.dim)
        new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
        
        cdef double old_pe = self.objective_function(self.molecules[index])
        cdef double new_pe = self.objective_function(new_solution)
        
        if old_pe + self.kinetic_energies[index] >= new_pe:
            self.molecules[index] = new_solution
            self.kinetic_energies[index] = old_pe + self.kinetic_energies[index] - new_pe
            return True
        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def decomposition(self, int index):
        """ Simulate decomposition reaction (global exploration) """
        cdef double old_pe = self.objective_function(self.molecules[index])
        if self.kinetic_energies[index] + old_pe >= self.alpha:
            split1 = self.molecules[index] + self.split_ratio * np.random.randn(self.dim)
            split2 = self.molecules[index] - self.split_ratio * np.random.randn(self.dim)
            split1 = np.clip(split1, self.bounds[:, 0], self.bounds[:, 1])
            split2 = np.clip(split2, self.bounds[:, 0], self.bounds[:, 1])
            
            pe1 = self.objective_function(split1)
            pe2 = self.objective_function(split2)
            
            if old_pe + self.kinetic_energies[index] >= pe1 + pe2:
                self.molecules[index] = split1
                self.kinetic_energies[index] = (old_pe + self.kinetic_energies[index] - pe1 - pe2) / 2
                self.molecules = np.vstack([self.molecules, split2])
                self.kinetic_energies = np.append(self.kinetic_energies, 
                                                 (old_pe + self.kinetic_energies[index] - pe1 - pe2) / 2)
                self.population_size += 1
                return True
        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def inter_molecular_collision(self, int index1, int index2):
        """ Simulate inter-molecular ineffective collision (local search) """
        cdef double r1 = np.random.rand()
        cdef double r2 = np.random.rand()
        new_solution1 = self.molecules[index1] + r1 * np.random.randn(self.dim)
        new_solution2 = self.molecules[index2] + r2 * np.random.randn(self.dim)
        new_solution1 = np.clip(new_solution1, self.bounds[:, 0], self.bounds[:, 1])
        new_solution2 = np.clip(new_solution2, self.bounds[:, 0], self.bounds[:, 1])
        
        old_pe1 = self.objective_function(self.molecules[index1])
        old_pe2 = self.objective_function(self.molecules[index2])
        new_pe1 = self.objective_function(new_solution1)
        new_pe2 = self.objective_function(new_solution2)
        
        if old_pe1 + old_pe2 + self.kinetic_energies[index1] + self.kinetic_energies[index2] >= new_pe1 + new_pe2:
            self.molecules[index1] = new_solution1
            self.molecules[index2] = new_solution2
            total_ke = old_pe1 + old_pe2 + self.kinetic_energies[index1] + self.kinetic_energies[index2] - new_pe1 - new_pe2
            self.kinetic_energies[index1] = total_ke * np.random.rand()
            self.kinetic_energies[index2] = total_ke - self.kinetic_energies[index1]
            return True
        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def synthesis(self, int index1, int index2):
        """ Simulate synthesis reaction (global exploration) """
        old_pe1 = self.objective_function(self.molecules[index1])
        old_pe2 = self.objective_function(self.molecules[index2])
        new_solution = (self.molecules[index1] + self.molecules[index2]) / 2
        new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
        new_pe = self.objective_function(new_solution)
        
        if old_pe1 + old_pe2 + self.kinetic_energies[index1] + self.kinetic_energies[index2] >= new_pe + self.beta:
            self.molecules[index1] = new_solution
            self.kinetic_energies[index1] = old_pe1 + old_pe2 + self.kinetic_energies[index1] + self.kinetic_energies[index2] - new_pe
            self.molecules = np.delete(self.molecules, index2, axis=0)
            self.kinetic_energies = np.delete(self.kinetic_energies, index2)
            self.population_size -= 1
            self.buffer += np.random.rand() * self.kinetic_energies[index1]
            return True
        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def elimination_phase(self):
        """ Eliminate worst molecules """
        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness = self.evaluate_molecules()
        cdef cnp.ndarray[cnp.npy_intp, ndim=1] worst_indices = np.argsort(fitness)[-int(self.elimination_ratio * self.population_size):]
        cdef int i
        for i in worst_indices:
            self.molecules[i] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
            self.kinetic_energies[i] = self.initial_ke
            self.buffer += np.random.rand() * self.kinetic_energies[i]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Chemical Reaction Optimization """
        self.initialize_molecules()
        cdef int generation, i, index1, index2, min_idx
        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness
        for generation in range(self.max_iter):
            fitness = self.evaluate_molecules()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.molecules[min_idx].copy()
                self.best_value = fitness[min_idx]

            i = 0
            while i < self.population_size:
                if np.random.rand() < self.mole_coll and i < self.population_size - 1:
                    index1 = i
                    index2 = i + 1
                    if np.random.rand() < 0.5:
                        self.inter_molecular_collision(index1, index2)
                    else:
                        if index1 < self.population_size and index2 < self.population_size:
                            self.synthesis(index1, index2)
                    i += 2
                else:
                    if i < self.population_size:
                        if np.random.rand() < 0.5:
                            self.on_wall_collision(i)
                        else:
                            self.decomposition(i)
                    i += 1

            self.elimination_phase()

            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
