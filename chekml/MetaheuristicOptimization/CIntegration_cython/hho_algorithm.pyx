# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from scipy.special import gamma
cimport cython
from libc.math cimport sin, pi, pow, abs as c_abs
from libc.stdlib cimport rand, RAND_MAX

# _____________________________________________________
# Main paper:
# Harris hawks optimization: Algorithm and applications
# Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
# Future Generation Computer Systems, 
# DOI: https://doi.org/10.1016/j.future.2019.02.028
# _____________________________________________________

cdef cnp.ndarray[cnp.float64_t, ndim=2] initialization(int population_size, int dim, object upper_bound, object lower_bound):
    """
    Initialize the population of Harris' hawks.

    Parameters:
    - population_size: Number of hawks.
    - dim: Number of dimensions.
    - upper_bound: Upper bound for each dimension (scalar or array).
    - lower_bound: Lower bound for each dimension (scalar or array).

    Returns:
    - X: Initial population (population_size x dim).
    """
    if np.isscalar(upper_bound):
        return np.random.uniform(lower_bound, upper_bound, (population_size, dim))
    else:
        X = np.zeros((population_size, dim), dtype=np.float64)
        for i in range(dim):
            X[:, i] = np.random.uniform(lower_bound[i], upper_bound[i], population_size)
        return X

cdef cnp.ndarray[cnp.float64_t, ndim=1] levy_flight(int dim):
    """
    Generate Levy flight step.

    Parameters:
    - dim: Number of dimensions.

    Returns:
    - step: Levy flight step vector.
    """
    cdef double beta = 1.5
    cdef double sigma = (gamma(1 + beta) * sin(pi * beta / 2) / 
                         (gamma((1 + beta) / 2) * beta * pow(2, (beta - 1) / 2))) ** (1 / beta)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] u = np.random.normal(0, sigma, dim)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] v = np.random.normal(0, 1, dim)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] step = np.zeros(dim, dtype=np.float64)
    cdef int i
    for i in range(dim):
        step[i] = u[i] / pow(c_abs(v[i]), 1 / beta)
    return step

def harris_hawks_optimization(int population_size, int max_iterations, object lower_bound, object upper_bound, int dim, object objective_function):
    """
    Harris Hawks Optimization (HHO) algorithm.

    Parameters:
    - population_size: Number of search agents (hawks).
    - max_iterations: Maximum number of iterations.
    - lower_bound: Lower bound for each dimension (scalar or array).
    - upper_bound: Upper bound for each dimension (scalar or array).
    - dim: Number of dimensions.
    - objective_function: Function to optimize.

    Returns:
    - Rabbit_Location: Best solution found.
    - Rabbit_Energy: Best fitness value.
    - CNVG: Convergence curve (fitness history).
    """
    print("HHO is now tackling your problem")
    
    # Initialize the location and energy of the rabbit
    cdef cnp.ndarray[cnp.float64_t, ndim=1] Rabbit_Location = np.zeros(dim, dtype=np.float64)
    cdef double Rabbit_Energy = float('inf')
    
    # Initialize the locations of Harris' hawks
    cdef cnp.ndarray[cnp.float64_t, ndim=2] X = initialization(population_size, dim, upper_bound, lower_bound)
    
    # Initialize convergence curve
    cdef cnp.ndarray[cnp.float64_t, ndim=1] CNVG = np.zeros(max_iterations, dtype=np.float64)
    
    cdef int t = 0
    cdef int i
    cdef double fitness, E1, E0, Escaping_Energy, q, r, Jump_strength
    cdef int rand_hawk_index
    cdef cnp.ndarray[cnp.float64_t, ndim=1] X_rand, X1, X2
    
    while t < max_iterations:
        # Check boundaries and evaluate fitness
        for i in range(population_size):
            # Clip to bounds
            X[i, :] = np.clip(X[i, :], lower_bound, upper_bound)
            # Compute fitness
            fitness = objective_function(X[i, :])
            # Update rabbit's location if better
            if fitness < Rabbit_Energy:
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()
        
        # Update energy factor
        E1 = 2 * (1 - (<double>t / max_iterations))
        
        # Update each hawk's position
        for i in range(population_size):
            E0 = 2 * (<double>rand() / RAND_MAX) - 1
            Escaping_Energy = E1 * E0
            
            if c_abs(Escaping_Energy) >= 1:
                # Exploration phase
                q = <double>rand() / RAND_MAX
                rand_hawk_index = <int>(<double>rand() / RAND_MAX * population_size)
                X_rand = X[rand_hawk_index, :]
                
                if q < 0.5:
                    X[i, :] = X_rand - (<double>rand() / RAND_MAX) * np.abs(X_rand - 2 * (<double>rand() / RAND_MAX) * X[i, :])
                else:
                    X[i, :] = (Rabbit_Location - np.mean(X, axis=0)) - \
                              (<double>rand() / RAND_MAX) * ((upper_bound - lower_bound) * (<double>rand() / RAND_MAX) + lower_bound)
            
            else:
                # Exploitation phase
                r = <double>rand() / RAND_MAX
                
                if r >= 0.5 and c_abs(Escaping_Energy) < 0.5:
                    X[i, :] = Rabbit_Location - Escaping_Energy * np.abs(Rabbit_Location - X[i, :])
                
                elif r >= 0.5 and c_abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - (<double>rand() / RAND_MAX))
                    X[i, :] = (Rabbit_Location - X[i, :]) - \
                              Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - X[i, :])
                
                elif r < 0.5 and c_abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - (<double>rand() / RAND_MAX))
                    X1 = Rabbit_Location - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - X[i, :])
                    
                    if objective_function(X1) < objective_function(X[i, :]):
                        X[i, :] = X1
                    else:
                        X2 = Rabbit_Location - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - X[i, :]) + \
                             np.random.rand(dim) * levy_flight(dim)
                        if objective_function(X2) < objective_function(X[i, :]):
                            X[i, :] = X2
                
                elif r < 0.5 and c_abs(Escaping_Energy) < 0.5:
                    Jump_strength = 2 * (1 - (<double>rand() / RAND_MAX))
                    X1 = Rabbit_Location - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - np.mean(X, axis=0))
                    
                    if objective_function(X1) < objective_function(X[i, :]):
                        X[i, :] = X1
                    else:
                        X2 = Rabbit_Location - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - np.mean(X, axis=0)) + \
                             np.random.rand(dim) * levy_flight(dim)
                        if objective_function(X2) < objective_function(X[i, :]):
                            X[i, :] = X2
        
        # Store convergence information
        CNVG[t] = Rabbit_Energy
        t += 1
    
    print(f"The best location of HHO is: {Rabbit_Location}")
    print(f"The best fitness of HHO is: {Rabbit_Energy}")
    
    return Rabbit_Location, Rabbit_Energy, CNVG
