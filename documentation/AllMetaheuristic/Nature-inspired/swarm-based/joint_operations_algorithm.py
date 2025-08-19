import numpy as np

class JointOperationsAlgorithm:
    def __init__(self, objective_function, num_variables, bounds, num_subpopulations=5, 
                 population_size_per_subpop=10, max_iterations=100, mutation_rate=0.1):
        """
        Initialize the Joint Operations Algorithm (JOA) optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - num_variables: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - num_subpopulations: Number of subpopulations.
        - population_size_per_subpop: Population size per subpopulation.
        - max_iterations: Maximum number of iterations.
        - mutation_rate: Rate of movement towards other individuals.
        """
        self.objective_function = objective_function
        self.num_variables = num_variables
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.num_subpopulations = num_subpopulations
        self.population_size_per_subpop = population_size_per_subpop
        self.max_iterations = max_iterations
        self.mutation_rate = mutation_rate

        self.populations = None  # List of subpopulation arrays
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_populations(self):
        """Generate random initial populations for each subpopulation."""
        self.populations = [
            np.random.uniform(
                self.bounds[:, 0], 
                self.bounds[:, 1], 
                (self.population_size_per_subpop, self.num_variables)
            ) for _ in range(self.num_subpopulations)
        ]

    def evaluate_populations(self):
        """Compute fitness values for all individuals in each subpopulation."""
        fitness = np.zeros((self.num_subpopulations, self.population_size_per_subpop))
        for i in range(self.num_subpopulations):
            for j in range(self.population_size_per_subpop):
                fitness[i, j] = self.objective_function(self.populations[i][j, :])
        return fitness

    def update_populations(self):
        """Update each subpopulation's position based on random interactions."""
        for i in range(self.num_subpopulations):
            for j in range(self.population_size_per_subpop):
                # Select a random subpopulation (excluding the current one)
                other_subpop_index = np.random.randint(0, self.num_subpopulations - 1)
                if other_subpop_index >= i:
                    other_subpop_index += 1

                # Select a random individual from the selected subpopulation
                other_individual_index = np.random.randint(0, self.population_size_per_subpop)

                # Move towards the selected individual
                direction = (self.populations[other_subpop_index][other_individual_index, :] - 
                            self.populations[i][j, :])
                self.populations[i][j, :] += self.mutation_rate * direction

                # Ensure the new position is within bounds
                self.populations[i][j, :] = np.clip(
                    self.populations[i][j, :], 
                    self.bounds[:, 0], 
                    self.bounds[:, 1]
                )

    def optimize(self):
        """Run the Joint Operations Algorithm optimization."""
        self.initialize_populations()
        for iteration in range(self.max_iterations):
            # Evaluate fitness for all subpopulations
            fitness = self.evaluate_populations()

            # Find the best solution in this iteration
            min_fitness = np.min(fitness)
            min_idx = np.unravel_index(np.argmin(fitness), fitness.shape)
            if min_fitness < self.best_value:
                self.best_value = min_fitness
                self.best_solution = self.populations[min_idx[0]][min_idx[1], :].copy()

            # Update populations
            self.update_populations()

            # Store history
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

# Example objective function (Sphere function)
def sphere_function(x):
    return np.sum(x ** 2)
