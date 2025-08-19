import numpy as np
from scipy.special import gamma  # Import gamma function from scipy.special

# _____________________________________________________
# Main paper:
# Harris hawks optimization: Algorithm and applications
# Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
# Future Generation Computer Systems, 
# DOI: https://doi.org/10.1016/j.future.2019.02.028
# _____________________________________________________

#  Author, inventor and programmer: Ali Asghar Heidari,
#  PhD research intern, Department of Computer Science, School of Computing, National University of Singapore, Singapore
#  Exceptionally Talented Ph. DC funded by Iran's National Elites Foundation (INEF), University of Tehran
#  03-03-2019

#  Researchgate: https://www.researchgate.net/profile/Ali_Asghar_Heidari

#  e-Mail: as_heidari@ut.ac.ir, aliasghar68@gmail.com,
#  e-Mail (Singapore): aliasgha@comp.nus.edu.sg, t0917038@u.nus.edu
# _____________________________________________________
#  Co-author and Advisor: Seyedali Mirjalili
#
#         e-Mail: ali.mirjalili@gmail.com
#                 seyedali.mirjalili@griffithuni.edu.au
#
#       Homepage: http://www.alimirjalili.com
# _____________________________________________________
#  Co-authors: Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, and Hui-Ling Chen

#       Homepage: http://www.evo-ml.com/2019/03/02/hho/
# _____________________________________________________

def initialization(population_size, dim, upper_bound, lower_bound):
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
        X = np.random.uniform(lower_bound, upper_bound, (population_size, dim))
    else:
        X = np.zeros((population_size, dim))
        for i in range(dim):
            X[:, i] = np.random.uniform(lower_bound[i], upper_bound[i], population_size)
    return X

def levy_flight(dim):
    """
    Generate Levy flight step.

    Parameters:
    - dim: Number of dimensions.

    Returns:
    - step: Levy flight step vector.
    """
    beta = 1.5
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    step = u / np.abs(v) ** (1 / beta)
    return step

def harris_hawks_optimization(population_size, max_iterations, lower_bound, upper_bound, dim, objective_function):
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
    Rabbit_Location = np.zeros(dim)
    Rabbit_Energy = float('inf')
    
    # Initialize the locations of Harris' hawks
    X = initialization(population_size, dim, upper_bound, lower_bound)
    
    # Initialize convergence curve
    CNVG = np.zeros(max_iterations)
    
    t = 0  # Iteration counter
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
        E1 = 2 * (1 - (t / max_iterations))  # Decreasing energy of rabbit
        
        # Update each hawk's position
        for i in range(population_size):
            E0 = 2 * np.random.rand() - 1  # Random energy between -1 and 1
            Escaping_Energy = E1 * E0  # Escaping energy of rabbit
            
            if abs(Escaping_Energy) >= 1:
                # Exploration phase
                q = np.random.rand()
                rand_hawk_index = np.random.randint(0, population_size)
                X_rand = X[rand_hawk_index, :]
                
                if q < 0.5:
                    # Perch based on other family members
                    X[i, :] = X_rand - np.random.rand() * abs(X_rand - 2 * np.random.rand() * X[i, :])
                else:
                    # Perch on a random tall tree
                    X[i, :] = (Rabbit_Location - np.mean(X, axis=0)) - \
                              np.random.rand() * ((upper_bound - lower_bound) * np.random.rand() + lower_bound)
            
            else:
                # Exploitation phase
                r = np.random.rand()
                
                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    # Hard besiege
                    X[i, :] = Rabbit_Location - Escaping_Energy * abs(Rabbit_Location - X[i, :])
                
                elif r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    # Soft besiege
                    Jump_strength = 2 * (1 - np.random.rand())
                    X[i, :] = (Rabbit_Location - X[i, :]) - \
                              Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                
                elif r < 0.5 and abs(Escaping_Energy) >= 0.5:
                    # Soft besiege with rapid dives
                    Jump_strength = 2 * (1 - np.random.rand())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                    
                    if objective_function(X1) < objective_function(X[i, :]):
                        X[i, :] = X1
                    else:
                        X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :]) + \
                             np.random.rand(dim) * levy_flight(dim)
                        if objective_function(X2) < objective_function(X[i, :]):
                            X[i, :] = X2
                
                elif r < 0.5 and abs(Escaping_Energy) < 0.5:
                    # Hard besiege with rapid dives
                    Jump_strength = 2 * (1 - np.random.rand())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - np.mean(X, axis=0))
                    
                    if objective_function(X1) < objective_function(X[i, :]):
                        X[i, :] = X1
                    else:
                        X2 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - np.mean(X, axis=0)) + \
                             np.random.rand(dim) * levy_flight(dim)
                        if objective_function(X2) < objective_function(X[i, :]):
                            X[i, :] = X2
        
        # Store convergence information
        CNVG[t] = Rabbit_Energy
        t += 1
        
        # Optional progress display
        # if t % 100 == 0:
        #     print(f"At iteration {t}, the best fitness is {Rabbit_Energy}")
    
    print(f"The best location of HHO is: {Rabbit_Location}")
    print(f"The best fitness of HHO is: {Rabbit_Energy}")
    
    return Rabbit_Location, Rabbit_Energy, CNVG

# Example usage with a test function
if __name__ == "__main__":
    # Define a simple test function (F1: Sphere function)
    def F1(x):
        return np.sum(x ** 2)
    
    # Parameters
    N = 30  # Number of search agents
    T = 500  # Maximum iterations
    lb = -100  # Lower bound
    ub = 100   # Upper bound
    dim = 30   # Dimensions
    
    # Run HHO
    best_location, best_fitness, convergence = harris_hawks_optimization(N, T, lb, ub, dim, F1)
    
    # Optional: Plot convergence curve
    import matplotlib.pyplot as plt
    plt.figure()
    plt.semilogy(convergence, 'b-', linewidth=4)
    plt.title('Convergence curve')
    plt.xlabel('Iteration')
    plt.ylabel('Best fitness obtained so far')
    plt.grid(False)
    plt.legend(['HHO'])
    plt.savefig('hho_convergence.png')
