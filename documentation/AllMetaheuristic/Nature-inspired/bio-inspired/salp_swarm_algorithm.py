import numpy as np

class SalpSwarmAlgorithm:
    """
    Salp Swarm Algorithm (SSA) for optimization problems.

    Parameters:
    - objective_function: Function to optimize.
    - dim: Number of dimensions (variables).
    - bounds: Tuple of (lower, upper) bounds for each dimension.
    - population_size: Number of salp search agents (default: 30).
    - max_iter: Maximum number of iterations (default: 100).
    """
    def __init__(self, objective_function, dim, bounds, population_size=30, max_iter=100):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        
        self.salp_positions = None  # Population of salps
        self.food_position = None   # Best solution (food source)
        self.food_fitness = float("inf")  # Best fitness value
        self.convergence_curve = []  # History of best fitness per iteration

    def initialize_salps(self):
        """ Initialize the first population of salps """
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        if len(lb) == 1:  # Single bound for all dimensions
            self.salp_positions = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        else:  # Different bounds for each dimension
            self.salp_positions = np.zeros((self.population_size, self.dim))
            for i in range(self.dim):
                self.salp_positions[:, i] = np.random.rand(self.population_size) * (ub[i] - lb[i]) + lb[i]

    def evaluate_salps(self):
        """ Compute fitness values for the salp positions """
        return np.array([self.objective_function(pos) for pos in self.salp_positions])

    def update_leader_salp(self, index, c1):
        """ Update leader salp position based on food source (Eq. 3.1) """
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        new_position = np.zeros(self.dim)
        for j in range(self.dim):
            c2 = np.random.rand()
            c3 = np.random.rand()
            if c3 < 0.5:
                new_position[j] = self.food_position[j] + c1 * ((ub[j] - lb[j]) * c2 + lb[j])
            else:
                new_position[j] = self.food_position[j] - c1 * ((ub[j] - lb[j]) * c2 + lb[j])
        return np.clip(new_position, lb, ub)

    def update_follower_salp(self, index):
        """ Update follower salp position (Eq. 3.4) """
        point1 = self.salp_positions[index - 1]
        point2 = self.salp_positions[index]
        return (point1 + point2) / 2

    def optimize(self):
        """ Run the Salp Swarm Algorithm """
        # Initialize salp positions
        self.initialize_salps()
        
        # Evaluate initial fitness
        salp_fitness = self.evaluate_salps()
        
        # Initialize food source (best solution)
        best_idx = np.argmin(salp_fitness)
        self.food_position = self.salp_positions[best_idx].copy()
        self.food_fitness = salp_fitness[best_idx]
        
        # Sort salps by fitness for initial leader-follower structure
        sorted_indices = np.argsort(salp_fitness)
        self.salp_positions = self.salp_positions[sorted_indices]
        
        # Main loop
        for l in range(1, self.max_iter + 1):
            # Update c1 coefficient (Eq. 3.2)
            c1 = 2 * np.exp(-((4 * l / self.max_iter) ** 2))
            
            # Update salp positions
            for i in range(self.population_size):
                if i < self.population_size / 2:  # Leader salps
                    self.salp_positions[i] = self.update_leader_salp(i, c1)
                else:  # Follower salps
                    self.salp_positions[i] = self.update_follower_salp(i)
            
            # Boundary checking
            lb, ub = self.bounds[:, 0], self.bounds[:, 1]
            for i in range(self.population_size):
                above_ub = self.salp_positions[i] > ub
                below_lb = self.salp_positions[i] < lb
                self.salp_positions[i] = (self.salp_positions[i] * (~(above_ub + below_lb)) +
                                         ub * above_ub + lb * below_lb)
            
            # Evaluate fitness and update food source
            salp_fitness = self.evaluate_salps()
            for i in range(self.population_size):
                if salp_fitness[i] < self.food_fitness:
                    self.food_position = self.salp_positions[i].copy()
                    self.food_fitness = salp_fitness[i]
            
            # Store convergence data
            self.convergence_curve.append(self.food_fitness)
            print(f"Iteration {l}: Best Fitness = {self.food_fitness}")
        
        return self.food_position, self.food_fitness, self.convergence_curve

# Example usage
if __name__ == "__main__":
    # Example objective function (Sphere function)
    def sphere_function(x):
        return np.sum(x ** 2)
    
    # Parameters
    dim = 10
    bounds = [(-100, 100)] * dim  # Bounds for each dimension
    population_size = 30
    max_iter = 1000
    
    # Initialize and run SSA
    ssa = SalpSwarmAlgorithm(sphere_function, dim, bounds, population_size, max_iter)
    best_position, best_fitness, convergence = ssa.optimize()
    
    print(f"Best Solution: {best_position}")
    print(f"Best Fitness: {best_fitness}")
