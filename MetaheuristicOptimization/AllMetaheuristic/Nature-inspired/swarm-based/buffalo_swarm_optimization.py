import numpy as np

def buffalo_swarm_optimization(objective_function, num_buffaloes, num_iterations, num_variables, lower_bound, upper_bound):
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
    - best_solution: Best solution found.
    - best_fitness: Best fitness value.
    - history: List of tuples (iteration, best_solution, best_fitness) for each iteration.
    """
    # Initialize buffalo population
    buffaloes = np.random.rand(num_buffaloes, num_variables) * (upper_bound - lower_bound) + lower_bound
    
    # Initialize best solution and best fitness
    best_solution = buffaloes[0, :].copy()
    best_fitness = objective_function(best_solution)
    
    # Initialize history
    history = [(0, best_solution.copy(), best_fitness)]
    
    # Main loop
    for iter in range(num_iterations):
        # Update each buffalo's position
        for i in range(num_buffaloes):
            # Calculate new position based on local search
            new_position = local_search_with_bounds(buffaloes[i, :], objective_function, lower_bound, upper_bound)
            
            # Update buffalo position
            buffaloes[i, :] = new_position
            
            # Update best solution and best fitness if needed
            fitness = objective_function(new_position)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = new_position.copy()
        
        # Store best solution for this iteration
        history.append((iter + 1, best_solution.copy(), best_fitness))
    
    return best_solution, best_fitness, history

def local_search_with_bounds(current_position, objective_function, lower_bound, upper_bound):
    """
    Local search function with bounds.

    Parameters:
    - current_position: Current position of the buffalo.
    - objective_function: Function to evaluate fitness.
    - lower_bound: Lower bound for each dimension.
    - upper_bound: Upper bound for each dimension.

    Returns:
    - new_position: New position after local search.
    """
    # Perturb current position
    perturbation = np.random.randn(len(current_position)) * 0.1  # Small random perturbation
    new_position = current_position + perturbation
    
    # Ensure new position is within bounds
    new_position = np.clip(new_position, lower_bound, upper_bound)
    
    return new_position
