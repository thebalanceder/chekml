import numpy as np
import time
import matplotlib.pyplot as plt

class LightningSearchAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=500, 
                 max_channel_time=10, energy_factor=2.05):
        """
        Initialize the Lightning Search Algorithm (LSA).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of lightning channels (solutions).
        - max_iter: Maximum number of iterations.
        - max_channel_time: Maximum time before channel elimination.
        - energy_factor: Initial energy factor for movement.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.max_channel_time = max_channel_time
        self.energy_factor = energy_factor

        self.channels = None  # Population of lightning channels (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []
        self.directions = None
        self.channel_time = 0

    def initialize_channels(self):
        """ Generate initial lightning channels randomly """
        self.channels = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                         (self.population_size, self.dim))
        self.directions = np.random.uniform(-1, 1, (1, self.dim))

    def evaluate_channels(self):
        """ Compute fitness values for the lightning channels """
        return np.array([self.objective_function(channel) for channel in self.channels])

    def update_channel_elimination(self, fitness):
        """ Eliminate the worst channel after max_channel_time iterations """
        self.channel_time += 1
        if self.channel_time >= self.max_channel_time:
            sorted_indices = np.argsort(fitness)
            self.channels[sorted_indices[-1]] = self.channels[sorted_indices[0]].copy()
            fitness[sorted_indices[-1]] = fitness[sorted_indices[0]]
            self.channel_time = 0
        return fitness

    def update_directions(self, best_channel, best_fitness):
        """ Update the direction of lightning movement """
        for d in range(self.dim):
            test_channel = best_channel.copy()
            test_channel[d] += self.directions[0, d] * 0.005 * (self.bounds[d, 1] - self.bounds[d, 0])
            test_fitness = self.objective_function(test_channel)
            if test_fitness < best_fitness:
                self.directions[0, d] = self.directions[0, d]
            else:
                self.directions[0, d] = -self.directions[0, d]

    def update_positions(self, fitness, best_index, worst_fitness, t):
        """ Update channel positions based on energy and distance """
        energy = self.energy_factor - 2 * np.exp(-5 * (self.max_iter - t) / self.max_iter)
        for i in range(self.population_size):
            dist = self.channels[i] - self.channels[best_index]
            temp_channel = np.zeros(self.dim)
            for d in range(self.dim):
                if np.array_equal(self.channels[i], self.channels[best_index]):
                    temp_channel[d] = self.channels[i, d] + self.directions[0, d] * abs(np.random.normal(0, energy))
                else:
                    if dist[d] < 0:
                        temp_channel[d] = self.channels[i, d] + np.random.exponential(abs(dist[d]))
                    else:
                        temp_channel[d] = self.channels[i, d] - np.random.exponential(dist[d])
                
                # Boundary check
                if temp_channel[d] > self.bounds[d, 1] or temp_channel[d] < self.bounds[d, 0]:
                    temp_channel[d] = np.random.uniform(self.bounds[d, 0], self.bounds[d, 1])

            temp_fitness = self.objective_function(temp_channel)
            if temp_fitness < fitness[i]:
                self.channels[i] = temp_channel
                fitness[i] = temp_fitness

                # Focking procedure
                if np.random.rand() < 0.01:
                    fock_channel = self.bounds[:, 0] + self.bounds[:, 1] - temp_channel
                    fock_fitness = self.objective_function(fock_channel)
                    if fock_fitness < fitness[i]:
                        self.channels[i] = fock_channel
                        fitness[i] = fock_fitness
        return fitness

    def optimize(self):
        """ Run the Lightning Search Algorithm """
        start_time = time.time()
        self.initialize_channels()

        for t in range(self.max_iter):
            fitness = self.evaluate_channels()
            min_idx = np.argmin(fitness)
            max_idx = np.argmax(fitness)

            if fitness[min_idx] < self.best_value:
                self.best_solution = self.channels[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Update channel elimination
            fitness = self.update_channel_elimination(fitness)

            # Update directions based on best channel
            self.update_directions(self.channels[min_idx], fitness[min_idx])

            # Update positions
            fitness = self.update_positions(fitness, min_idx, fitness[max_idx], t)

            # Stop if best and worst are equal
            if np.abs(fitness[min_idx] - fitness[max_idx]) < 1e-10:
                break

            self.history.append((t, self.best_solution.copy(), self.best_value))

        elapsed_time = time.time() - start_time

        # Plot convergence
        iterations, _, values = zip(*self.history)
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, values, 'b-', linewidth=2)
        plt.xlabel('No of Iteration')
        plt.ylabel('Fitness Value')
        plt.title('Convergence of Lightning Search Algorithm')
        plt.grid(True)
        plt.savefig('lsa_convergence.png')

        print(f"Optimal value = {self.best_value}")
        return self.best_solution, self.best_value, self.history, elapsed_time
