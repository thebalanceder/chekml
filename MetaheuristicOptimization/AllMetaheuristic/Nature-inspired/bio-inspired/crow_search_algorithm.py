import numpy as np

class CrowSearchAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=20, max_iter=5000, 
                 awareness_probability=0.1, flight_length=2.0):
        """
        Initialize the Crow Search Algorithm (CSA) optimizer.

        Parameters:
        - objective_function: Function to optimize (minimization).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of crows (solutions).
        - max_iter: Maximum number of iterations.
        - awareness_probability: Probability of crow awareness (AP).
        - flight_length: Flight length parameter (fl).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.awareness_probability = awareness_probability
        self.flight_length = flight_length

        self.crows = None  # Population of crow positions
        self.memory = None  # Memory of best positions
        self.fitness_memory = None  # Fitness of memory positions
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_crows(self):
        """ Generate initial crow positions randomly """
        lower_bound, upper_bound = self.bounds[:, 0], self.bounds[:, 1]
        self.crows = lower_bound + (upper_bound - lower_bound) * np.random.rand(self.population_size, self.dim)
        self.memory = self.crows.copy()
        self.fitness_memory = self.evaluate_crows()

    def evaluate_crows(self):
        """ Compute fitness values for the crow positions """
        return np.array([self.objective_function(crow) for crow in self.crows])

    def update_positions(self):
        """ Update crow positions based on CSA rules """
        new_crows = self.crows.copy()
        random_crows = np.random.choice(self.population_size, self.population_size).astype(int)

        for i in range(self.population_size):
            if np.random.rand() > self.awareness_probability:
                # State 1: Follow another crow
                new_crows[i, :] = self.crows[i, :] + self.flight_length * np.random.rand() * \
                                  (self.memory[random_crows[i], :] - self.crows[i, :])
            else:
                # State 2: Random position within bounds
                new_crows[i, :] = self.bounds[:, 0] + (self.bounds[:, 1] - self.bounds[:, 0]) * np.random.rand(self.dim)

        # Ensure new positions are within bounds
        new_crows = np.clip(new_crows, self.bounds[:, 0], self.bounds[:, 1])
        return new_crows

    def update_memory(self, new_crows, new_fitness):
        """ Update crow positions and memory based on fitness """
        for i in range(self.population_size):
            # Check if new position is within bounds
            if np.all(new_crows[i, :] >= self.bounds[:, 0]) and np.all(new_crows[i, :] <= self.bounds[:, 1]):
                self.crows[i, :] = new_crows[i, :]  # Update position
                if new_fitness[i] < self.fitness_memory[i]:
                    self.memory[i, :] = new_crows[i, :]  # Update memory
                    self.fitness_memory[i] = new_fitness[i]

    def optimize(self):
        """ Run the Crow Search Algorithm """
        self.initialize_crows()

        for iteration in range(self.max_iter):
            # Update positions
            new_crows = self.update_positions()
            new_fitness = np.array([self.objective_function(crow) for crow in new_crows])

            # Update positions and memory
            self.update_memory(new_crows, new_fitness)

            # Track best solution
            min_fitness_idx = np.argmin(self.fitness_memory)
            if self.fitness_memory[min_fitness_idx] < self.best_value:
                self.best_solution = self.memory[min_fitness_idx].copy()
                self.best_value = self.fitness_memory[min_fitness_idx]

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

# Example usage with Sphere function
if __name__ == "__main__":
    def sphere_function(x):
        return np.sum(x ** 2)

    dim = 10
    bounds = np.array([(-100, 100)] * dim)
    csa = CrowSearchAlgorithm(
        objective_function=sphere_function,
        dim=dim,
        bounds=bounds,
        population_size=20,
        max_iter=5000,
        awareness_probability=0.1,
        flight_length=2.0
    )
    best_solution, best_value, history = csa.optimize()
    print(f"Best Solution: {best_solution}")
    print(f"Best Value: {best_value}")
