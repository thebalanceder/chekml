import numpy as np

class InvasiveWeedOptimization:
    def __init__(self, objective_function, dim, bounds, max_iter=200, 
                 initial_pop_size=10, max_pop_size=25, min_seeds=1, 
                 max_seeds=5, exponent=2, sigma_initial=0.5, sigma_final=0.001):
        """
        Initialize the Invasive Weed Optimization (IWO) algorithm.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - max_iter: Maximum number of iterations.
        - initial_pop_size: Initial population size.
        - max_pop_size: Maximum population size.
        - min_seeds: Minimum number of seeds per plant.
        - max_seeds: Maximum number of seeds per plant.
        - exponent: Variance reduction exponent.
        - sigma_initial: Initial standard deviation.
        - sigma_final: Final standard deviation.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.max_iter = max_iter
        self.initial_pop_size = initial_pop_size
        self.max_pop_size = max_pop_size
        self.min_seeds = min_seeds
        self.max_seeds = max_seeds
        self.exponent = exponent
        self.sigma_initial = sigma_initial
        self.sigma_final = sigma_final

        self.population = None  # Population of plants (solutions)
        self.best_solution = None
        self.best_cost = float("inf")
        self.history = []  # Renamed from cost_history

    def initialize_population(self):
        """ Generate initial population of plants randomly """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.initial_pop_size, self.dim))

    def evaluate_population(self):
        """ Compute fitness values for the population """
        return np.array([self.objective_function(plant) for plant in self.population])

    def update_standard_deviation(self, iteration):
        """ Update standard deviation based on iteration """
        return ((self.max_iter - iteration) / (self.max_iter - 1)) ** self.exponent * \
               (self.sigma_initial - self.sigma_final) + self.sigma_final

    def reproduction(self, costs, sigma):
        """ Generate seeds based on fitness values """
        best_cost = np.min(costs)
        worst_cost = np.max(costs)
        new_population = []

        for i, plant in enumerate(self.population):
            # Calculate number of seeds based on fitness
            if best_cost == worst_cost:
                num_seeds = self.min_seeds  # Ensure at least min_seeds
            else:
                ratio = (costs[i] - worst_cost) / (best_cost - worst_cost)
                num_seeds = int(self.min_seeds + (self.max_seeds - self.min_seeds) * ratio)

            # Generate seeds
            for _ in range(num_seeds):
                new_plant = plant + sigma * np.random.randn(self.dim)
                new_plant = np.clip(new_plant, self.bounds[:, 0], self.bounds[:, 1])
                new_population.append(new_plant)

        return np.array(new_population) if new_population else self.population.copy()

    def optimize(self):
        """ Run the Invasive Weed Optimization algorithm """
        self.initialize_population()

        for iteration in range(self.max_iter):
            # Evaluate current population
            costs = self.evaluate_population()
            min_idx = np.argmin(costs)
            if costs[min_idx] < self.best_cost:
                self.best_solution = self.population[min_idx].copy()
                self.best_cost = costs[min_idx]

            # Update standard deviation
            sigma = self.update_standard_deviation(iteration)

            # Reproduction phase
            new_population = self.reproduction(costs, sigma)

            # Merge populations only if new_population is not empty
            if new_population.size > 0:
                self.population = np.vstack([self.population, new_population])
            else:
                print(f"Iteration {iteration + 1}: No new seeds generated, keeping current population")

            # Sort population by cost
            costs = self.evaluate_population()
            sort_order = np.argsort(costs)
            self.population = self.population[sort_order]

            # Competitive exclusion
            if len(self.population) > self.max_pop_size:
                self.population = self.population[:self.max_pop_size]

            # Store best cost for this iteration
            self.history.append((iteration, self.best_solution.copy(), self.best_cost))

            # Display iteration information
            print(f"Iteration {iteration + 1}: Best Cost = {self.best_cost}")

        return self.best_solution, self.best_cost, self.history

# Example usage
if __name__ == "__main__":
    def sphere(x):
        """ Sphere function for testing """
        return np.sum(x ** 2)

    # Problem definition
    dim = 5
    bounds = [(-10, 10)] * dim

    # Initialize and run IWO
    iwo = InvasiveWeedOptimization(sphere, dim, bounds)
    best_solution, best_cost, history = iwo.optimize()

    # Plot results
    import matplotlib.pyplot as plt
    plt.semilogy([h[2] for h in history], 'LineWidth', 2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Cost')
    plt.grid(True)
    plt.savefig('iwo_results.png')
