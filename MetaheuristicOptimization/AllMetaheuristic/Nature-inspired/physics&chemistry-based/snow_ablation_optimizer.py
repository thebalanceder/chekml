import numpy as np
import random

# Optimization parameters
NUM_POP = 50  # Increased population size for better exploration
MAX_ITER = 1000  # Increased iterations for better convergence
LOW = -5  # Lower bound of solution space
UP = 5  # Upper bound of solution space
DF_MIN = 0.35  # Minimum degree day factor
DF_MAX = 0.6  # Maximum degree day factor

def initialize_population(dim, num_pop):
    """Initialize population randomly within bounds."""
    return LOW + np.random.rand(num_pop, dim) * (UP - LOW)

def brownian_motion(num_pop, dim):
    """Simulate Brownian motion with Gaussian distribution."""
    return np.random.normal(0, 0.5, (num_pop, dim))  # Reduced variance for stability

def calculate_centroid(population):
    """Calculate centroid of the population."""
    return np.mean(population, axis=0)

def select_elite(population, fitness, num_elites=4):
    """Select elite individuals based on fitness with rank-based probabilities."""
    indices = np.argsort(fitness)[:num_elites]
    ranks = np.array([4, 3, 2, 1])  # Higher rank for better individuals
    probs = ranks / np.sum(ranks)
    choice = np.random.choice(indices, p=probs)
    return population[choice], indices[0]  # Return selected elite and best index

def covariance_matrix_learning(population):
    """Apply covariance matrix learning strategy."""
    mean = np.mean(population, axis=0)
    cov_matrix = np.cov(population.T, bias=False)
    # Ensure positive semi-definite matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix + 1e-6 * np.eye(len(mean)))
    Q = eigenvectors  # Orthogonal matrix
    QQ = population @ Q
    return QQ

def historical_boundary_adjustment(position, dim):
    """Strictly enforce boundary constraints by clipping."""
    return np.clip(position, LOW, UP)

def random_centroid_reverse_learning(population, num_pop, dim):
    """Apply random centroid reverse learning strategy."""
    B = random.randint(2, num_pop // 2)  # Smaller subset for diversity
    indices = np.random.choice(num_pop, B, replace=False)
    selected = population[indices]
    centroid = np.mean(selected, axis=0)
    reverse_pop = 2 * centroid - population
    return np.clip(reverse_pop, LOW, UP)  # Enforce bounds immediately

def calculate_snow_ablation_rate(iter, max_iter):
    """Calculate snow ablation rate using degree day method."""
    T = np.exp(-iter / max_iter)  # Adjusted for gradual decay
    Df = DF_MIN + (DF_MAX - DF_MIN) * (np.exp(iter / max_iter) - 1) / (np.e - 1)
    return Df * T

def enforce_bound_constraints(population, dim):
    """Ensure population stays within bounds by clipping."""
    return np.clip(population, LOW, UP)

def MESAO_optimize(objective_function, dim, num_pop=NUM_POP, max_iter=MAX_ITER):
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
    random.seed(42)
    
    # Initialize population
    population = initialize_population(dim, num_pop)
    fitness = np.array([objective_function(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    history = [(0, best_solution.copy(), best_fitness)]
    
    # Initialize subpopulations
    num_a = num_pop // 2  # Exploration subpopulation
    num_b = num_pop - num_a  # Development subpopulation
    
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
            alpha1 = random.uniform(0, 1)
            population[idx] = elite + R * Bu * (
                alpha1 * (best_solution - population[idx]) +
                (1 - alpha1) * (centroid - population[idx])
            )
            population[idx] = historical_boundary_adjustment(population[idx], dim)
        
        # Development phase
        for idx in pop_b_indices:
            QQ = covariance_matrix_learning(population)
            population[idx] += R * QQ[idx]  # Scale by ablation rate
            population[idx] = historical_boundary_adjustment(population[idx], dim)
        
        # Boundary adjustment
        population = enforce_bound_constraints(population, dim)
        
        # Random centroid reverse learning
        reverse_pop = random_centroid_reverse_learning(population, num_pop, dim)
        reverse_fitness = np.array([objective_function(ind) for ind in reverse_pop])
        
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

# Example usage
if __name__ == "__main__":
    def sphere_function(x):
        """Test objective function: Sphere function."""
        return np.sum(x ** 2)
    
    dim = 2
    best_solution, best_fitness, history = MESAO_optimize(sphere_function, dim)
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
