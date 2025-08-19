import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX

# Ensure NumPy C API is initialized
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TeachingLearningBasedOptimizer:
    cdef object objective_function
    cdef int dim
    cdef double[:, :] bounds
    cdef int population_size
    cdef int max_iter
    cdef double[:, :] population
    cdef double[:] best_solution
    cdef double best_value
    cdef list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100):
        """
        Initialize the TLBO optimizer.

        Parameters:
        - objective_function: Function to optimize (minimization problem).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of students (solutions) in the population.
        - max_iter: Maximum number of iterations (generations).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)  # Convert to NumPy array for memory view
        self.population_size = population_size
        self.max_iter = max_iter
        self.population = None
        self.best_solution = None
        self.best_value = np.inf
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_population(self):
        """ Generate initial population randomly within bounds """
        cdef double[:] lb = self.bounds[:, 0]
        cdef double[:] ub = self.bounds[:, 1]
        cdef int i, j
        self.population = np.empty((self.population_size, self.dim), dtype=np.double)
        cdef double[:, :] pop = self.population
        for i in range(self.population_size):
            for j in range(self.dim):
                pop[i, j] = lb[j] + (ub[j] - lb[j]) * (rand() / <double>RAND_MAX)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] evaluate_population(self):
        """ Compute fitness values for the population """
        cdef double[:] fitness = np.empty(self.population_size, dtype=np.double)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.population[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] teacher_phase(self, int index, double[:] fitness):
        """ Simulate the teacher phase for the given student index """
        cdef double[:] mean_student = np.mean(self.population, axis=0)
        cdef int best_idx = np.argmin(fitness)
        cdef double[:] teacher = self.population[best_idx]
        cdef int tf = 1 + (rand() % 2)  # Random teaching factor (1 or 2)
        cdef double[:] new_solution = np.empty(self.dim, dtype=np.double)
        cdef double new_fitness
        cdef int j
        cdef double[:] lb = self.bounds[:, 0]
        cdef double[:] ub = self.bounds[:, 1]

        # Generate new solution
        for j in range(self.dim):
            new_solution[j] = self.population[index, j] + (rand() / <double>RAND_MAX) * (teacher[j] - tf * mean_student[j])
            # Bound the solution
            if new_solution[j] < lb[j]:
                new_solution[j] = lb[j]
            elif new_solution[j] > ub[j]:
                new_solution[j] = ub[j]

        # Evaluate new solution
        new_fitness = self.objective_function(new_solution)

        # Greedy selection
        if new_fitness < fitness[index]:
            for j in range(self.dim):
                self.population[index, j] = new_solution[j]
            fitness[index] = new_fitness

        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] learner_phase(self, int index, double[:] fitness, int partner_idx):
        """ Simulate the learner phase for the given student index """
        cdef double[:] new_solution = np.empty(self.dim, dtype=np.double)
        cdef double new_fitness
        cdef int j
        cdef double[:] lb = self.bounds[:, 0]
        cdef double[:] ub = self.bounds[:, 1]

        # Generate new solution
        if fitness[index] < fitness[partner_idx]:
            for j in range(self.dim):
                new_solution[j] = self.population[index, j] + (rand() / <double>RAND_MAX) * (self.population[index, j] - self.population[partner_idx, j])
        else:
            for j in range(self.dim):
                new_solution[j] = self.population[index, j] + (rand() / <double>RAND_MAX) * (self.population[partner_idx, j] - self.population[index, j])

        # Bound the solution
        for j in range(self.dim):
            if new_solution[j] < lb[j]:
                new_solution[j] = lb[j]
            elif new_solution[j] > ub[j]:
                new_solution[j] = ub[j]

        # Evaluate new solution
        new_fitness = self.objective_function(new_solution)

        # Greedy selection
        if new_fitness < fitness[index]:
            for j in range(self.dim):
                self.population[index, j] = new_solution[j]
            fitness[index] = new_fitness

        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Teaching-Learning-Based Optimization """
        # Initialize population
        self.initialize_population()
        
        # Evaluate initial population
        cdef double[:] fitness = self.evaluate_population()
        
        cdef int generation, i, min_idx
        cdef double current_best_value
        cdef int[:] partners
        
        for generation in range(self.max_iter):
            # Partner selection using random permutation
            partners = np.random.permutation(self.population_size).astype(np.int32)
            
            # Process each student
            for i in range(self.population_size):
                # Teacher phase
                fitness = self.teacher_phase(i, fitness)
                
                # Learner phase
                fitness = self.learner_phase(i, fitness, partners[i])
            
            # Track best solution and fitness
            min_idx = np.argmin(fitness)
            current_best_value = fitness[min_idx]
            if current_best_value < self.best_value:
                self.best_solution = self.population[min_idx].copy()
                self.best_value = current_best_value
            
            # Store history
            self.history.append((generation, np.asarray(self.best_solution).copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")
        
        return np.asarray(self.best_solution), self.best_value, self.history
