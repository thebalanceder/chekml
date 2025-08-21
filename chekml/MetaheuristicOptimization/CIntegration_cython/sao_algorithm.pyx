# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sin, sqrt, cos, fabs
from libc.stdlib cimport rand, RAND_MAX
from cython.parallel import prange

# Define types for NumPy arrays
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# Optimization parameters
cdef int NUM_POP = 50
cdef int MAX_ITER = 1000
cdef double LOW = -5.0
cdef double UP = 5.0
cdef double DF_MIN = 0.35
cdef double DF_MAX = 0.6

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=2] initialize_population(int dim, int num_pop):
    """Initialize population randomly within bounds."""
    return LOW + np.random.rand(num_pop, dim) * (UP - LOW)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=2] brownian_motion(int num_pop, int dim):
    """Simulate Brownian motion with Gaussian distribution."""
    return np.random.normal(0, 0.5, (num_pop, dim))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=1] calculate_centroid(np.ndarray[DTYPE_t, ndim=2] population):
    """Calculate centroid of the population."""
    return np.mean(population, axis=0)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple select_elite(np.ndarray[DTYPE_t, ndim=2] population, np.ndarray[DTYPE_t, ndim=1] fitness, int num_elites=4):
    """Select elite individuals based on fitness with rank-based probabilities."""
    cdef np.ndarray[long] indices = np.argsort(fitness)[:num_elites]
    cdef np.ndarray[DTYPE_t, ndim=1] ranks = np.array([4.0, 3.0, 2.0, 1.0], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] probs = ranks / np.sum(ranks)
    cdef int choice = np.random.choice(indices, p=probs)
    return population[choice], indices[0]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=2] covariance_matrix_learning(np.ndarray[DTYPE_t, ndim=2] population):
    """Apply covariance matrix learning strategy."""
    cdef np.ndarray[DTYPE_t, ndim=1] mean = np.mean(population, axis=0)
    cdef np.ndarray[DTYPE_t, ndim=2] cov_matrix = np.cov(population.T, bias=False)
    cov_matrix += 1e-6 * np.eye(len(mean))
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    cdef np.ndarray[DTYPE_t, ndim=2] Q = eigenvectors
    return population @ Q

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=1] historical_boundary_adjustment(np.ndarray[DTYPE_t, ndim=1] position):
    """Strictly enforce boundary constraints by clipping."""
    return np.clip(position, LOW, UP)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=2] random_centroid_reverse_learning(np.ndarray[DTYPE_t, ndim=2] population, int num_pop, int dim):
    """Apply random centroid reverse learning strategy."""
    cdef int B = 2 + (rand() % ((num_pop // 2) - 1))  # Random int in [2, num_pop//2]
    cdef np.ndarray[long] indices = np.random.choice(num_pop, B, replace=False)
    cdef np.ndarray[DTYPE_t, ndim=2] selected = population[indices]
    cdef np.ndarray[DTYPE_t, ndim=1] centroid = np.mean(selected, axis=0)
    cdef np.ndarray[DTYPE_t, ndim=2] reverse_pop = 2 * centroid - population
    return np.clip(reverse_pop, LOW, UP)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double calculate_snow_ablation_rate(int iter, int max_iter):
    """Calculate snow ablation rate using degree day method."""
    cdef double T = exp(-iter / <double>max_iter)
    cdef double Df = DF_MIN + (DF_MAX - DF_MIN) * (exp(iter / <double>max_iter) - 1) / (exp(1) - 1)
    return Df * T

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=2] enforce_bound_constraints(np.ndarray[DTYPE_t, ndim=2] population):
    """Ensure population stays within bounds by clipping."""
    return np.clip(population, LOW, UP)

@cython.boundscheck(False)
@cython.wraparound(False)
def MESAO_optimize(objective_function, int dim, int num_pop=NUM_POP, int max_iter=MAX_ITER):
    """
    Main MESAO optimization function.
    
    Args:
        objective_function: Function to minimize
        dim: Dimension of the solution space
        num_pop: Population size
        max_iter: Maximum number of iterations
    
    Returns:
        best_solution: Best position found
        best_fitness: Best fitness value
        history: List of (iteration, best_solution, best_fitness)
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize population
    cdef np.ndarray[DTYPE_t, ndim=2] population = initialize_population(dim, num_pop)
    cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.array([objective_function(ind) for ind in population], dtype=DTYPE)
    cdef int best_idx = np.argmin(fitness)
    cdef np.ndarray[DTYPE_t, ndim=1] best_solution = population[best_idx].copy()
    cdef double best_fitness = fitness[best_idx]
    history = [(0, best_solution.copy(), best_fitness)]
    
    # Initialize subpopulations
    cdef int num_a = num_pop // 2
    cdef int num_b = num_pop - num_a
    cdef int iter, idx, i
    cdef np.ndarray[long] indices
    cdef np.ndarray[DTYPE_t, ndim=1] elite
    cdef np.ndarray[DTYPE_t, ndim=1] centroid
    cdef np.ndarray[DTYPE_t, ndim=1] Bu
    cdef double alpha1, R
    cdef np.ndarray[DTYPE_t, ndim=2] QQ
    cdef np.ndarray[DTYPE_t, ndim=2] reverse_pop
    cdef np.ndarray[DTYPE_t, ndim=1] reverse_fitness
    cdef np.ndarray[DTYPE_t, ndim=2] combined_pop
    cdef np.ndarray[DTYPE_t, ndim=1] combined_fitness
    cdef int min_fitness_idx
    
    for iter in range(max_iter):
        # Calculate snow ablation rate
        R = calculate_snow_ablation_rate(iter, max_iter)
        
        # Randomly split population
        indices = np.random.permutation(num_pop)
        pop_a_indices = indices[:num_a]
        pop_b_indices = indices[num_a:]
        
        # Exploration phase
        for idx in pop_a_indices:
            elite, _ = select_elite(population, fitness)
            centroid = calculate_centroid(population)
            Bu = brownian_motion(1, dim)[0]
            alpha1 = rand() / <double>RAND_MAX
            population[idx] = elite + R * Bu * (
                alpha1 * (best_solution - population[idx]) +
                (1 - alpha1) * (centroid - population[idx])
            )
            population[idx] = historical_boundary_adjustment(population[idx])
        
        # Development phase
        for idx in pop_b_indices:
            QQ = covariance_matrix_learning(population)
            population[idx] += R * QQ[idx]
            population[idx] = historical_boundary_adjustment(population[idx])
        
        # Boundary adjustment
        population = enforce_bound_constraints(population)
        
        # Random centroid reverse learning
        reverse_pop = random_centroid_reverse_learning(population, num_pop, dim)
        reverse_fitness = np.array([objective_function(ind) for ind in reverse_pop], dtype=DTYPE)
        
        # Greedy selection
        combined_pop = np.vstack((population, reverse_pop))
        combined_fitness = np.hstack((fitness, reverse_fitness))
        indices = np.argsort(combined_fitness)[:num_pop]
        population = combined_pop[indices]
        fitness = combined_fitness[indices]
        
        # Update best solution
        min_fitness_idx = np.argmin(fitness)
        if fitness[min_fitness_idx] < best_fitness:
            best_fitness = fitness[min_fitness_idx]
            best_solution = population[min_fitness_idx].copy()
            history.append((iter + 1, best_solution.copy(), best_fitness))
        
        # Adjust subpopulation sizes
        if num_a < num_pop:
            num_a += 1
            num_b -= 1
    
    return best_solution, best_fitness, history
