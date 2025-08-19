import numpy as np

class ThermalExchangeOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 step_size=0.1, initial_temperature=100, final_temperature=0.01, cooling_rate=0.99):
        """
        Initialize the Thermal Exchange Optimizer (TEO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of solutions.
        - max_iter: Maximum number of iterations.
        - step_size: Step size for solution perturbation.
        - initial_temperature: Initial temperature for annealing process.
        - final_temperature: Final temperature for convergence check.
        - cooling_rate: Cooling rate for temperature reduction.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate

        self.current_solution = None
        self.current_fitness = None
        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    def initialize_solution(self):
        """ Generate initial random solution """
        self.current_solution = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
        self.current_fitness = self.objective_function(self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_fitness = self.current_fitness

    def perturb_solution(self):
        """ Generate a new solution by perturbing the current solution """
        new_solution = self.current_solution + self.step_size * np.random.randn(self.dim)
        return np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

    def accept_solution(self, new_solution, new_fitness, temperature):
        """ Determine whether to accept the new solution based on fitness and temperature """
        if new_fitness < self.current_fitness or np.random.rand() < np.exp((self.current_fitness - new_fitness) / temperature):
            self.current_solution = new_solution
            self.current_fitness = new_fitness

    def update_best_solution(self):
        """ Update the best solution if the current solution is better """
        if self.current_fitness < self.best_fitness:
            self.best_solution = self.current_solution.copy()
            self.best_fitness = self.current_fitness

    def optimize(self):
        """ Run the Thermal Exchange Optimization algorithm """
        self.initialize_solution()
        temperature = self.initial_temperature

        for iteration in range(self.max_iter):
            # Generate and evaluate new solution
            new_solution = self.perturb_solution()
            new_fitness = self.objective_function(new_solution)

            # Accept or reject the new solution
            self.accept_solution(new_solution, new_fitness, temperature)

            # Update the best solution
            self.update_best_solution()

            # Reduce temperature
            temperature *= self.cooling_rate

            # Store history
            self.history.append((iteration, self.best_solution.copy(), self.best_fitness))
            print(f"Iteration {iteration + 1}: Best Fitness = {self.best_fitness}")

            # Check for convergence
            if temperature < self.final_temperature:
                break

        return self.best_solution, self.best_fitness, self.history

# Example objective function (Sphere function)
def sphere_function(x):
    return np.sum(x**2)

# Example usage
if __name__ == "__main__":
    dim = 10
    bounds = [(-5, 5)] * dim
    teo = ThermalExchangeOptimizer(sphere_function, dim, bounds)
    best_solution, best_fitness, history = teo.optimize()
    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness}")
