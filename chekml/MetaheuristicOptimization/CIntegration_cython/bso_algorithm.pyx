# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt

# Define types for NumPy arrays
ctypedef np.double_t DTYPE_t

# Define a type for the objective function (Python callable)
ctypedef double (*obj_func_t)(np.ndarray)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:, :] initialize_buffaloes(int num_buffaloes, int num_variables, double lower_bound, double upper_bound):
    """Initialize buffalo population with random positions."""
    cdef double[:, :] buffaloes = np.random.rand(num_buffaloes, num_variables)
    cdef int i, j
    cdef double scale = upper_bound - lower_bound
    for i in range(num_buffaloes):
        for j in range(num_variables):
            buffaloes[i, j] = buffaloes[i, j] * scale + lower_bound
    return buffaloes

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:] local_search_with_bounds(double[:] current_position, object objective_function,
                                       double lower_bound, double upper_bound, int num_variables):
    """Perform local search with bounds."""
    cdef double[:] new_position = np.zeros(num_variables, dtype=np.double)
    cdef double[:] perturbation = np.random.randn(num_variables)
    cdef int i
    cdef double pert_scale = 0.1  # Perturbation magnitude

    # Perturb current position
    for i in range(num_variables):
        new_position[i] = current_position[i] + perturbation[i] * pert_scale

        # Ensure new position is within bounds
        if new_position[i] < lower_bound:
            new_position[i] = lower_bound
        elif new_position[i] > upper_bound:
            new_position[i] = upper_bound

    return new_position

@cython.boundscheck(False)
@cython.wraparound(False)
def buffalo_swarm_optimization(object objective_function, int num_buffaloes, int num_iterations,
                              int num_variables, double lower_bound, double upper_bound):
    """
    Buffalo Swarm Optimization (BSO) for a given objective function.

    Parameters:
    - objective_function: Function to optimize (takes a numpy array and returns a scalar).
    - num_buffaloes: Number of buffaloes in the population.
    - num_iterations: Maximum number of iterations.
    - num_variables: Number of dimensions (variables).
    - lower_bound: Lower bound for each dimension.
    - upper_bound: Upper bound for each dimension.

    Returns:
    - best_solution: Best solution found (numpy array).
    - best_fitness: Best fitness value (float).
    - history: List of tuples (iteration, best_solution, best_fitness) for each iteration.
    """
    # Initialize buffalo population
    cdef double[:, :] buffaloes = initialize_buffaloes(num_buffaloes, num_variables, lower_bound, upper_bound)
    
    # Initialize best solution and best fitness
    cdef double[:] best_solution = np.array(buffaloes[0, :], dtype=np.double)
    cdef double best_fitness = objective_function(np.asarray(best_solution))
    
    # Initialize history
    history = [(0, np.asarray(best_solution).copy(), best_fitness)]
    
    # Main loop
    cdef int iter, i
    cdef double[:] new_position
    cdef double fitness
    for iter in range(num_iterations):
        # Update each buffalo's position
        for i in range(num_buffaloes):
            # Calculate new position based on local search
            new_position = local_search_with_bounds(buffaloes[i, :], objective_function,
                                                  lower_bound, upper_bound, num_variables)
            
            # Update buffalo position
            for j in range(num_variables):
                buffaloes[i, j] = new_position[j]
            
            # Update best solution and best fitness if needed
            fitness = objective_function(np.asarray(new_position))
            if fitness < best_fitness:
                best_fitness = fitness
                for j in range(num_variables):
                    best_solution[j] = new_position[j]
        
        # Store best solution for this iteration
        history.append((iter + 1, np.asarray(best_solution).copy(), best_fitness))
    
    return np.asarray(best_solution), best_fitness, history
