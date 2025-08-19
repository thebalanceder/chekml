import numpy as np
import random

class VariableNeighborhoodSearch:
    def __init__(self, objective_function, dim, bounds, max_iterations=100, 
                 neighborhood_sizes=[1, 2, 3], mutation_rate=0.1):
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.neighborhood_sizes = neighborhood_sizes
        self.num_neighborhoods = len(neighborhood_sizes)  # ← FIXED HERE
        self.mutation_rate = mutation_rate


    def _generate_neighbor(self, current_solution, neighborhood_size):
        mutation = self.mutation_rate * (np.random.rand(self.dim) - 0.5) * neighborhood_size
        neighbor = np.array(current_solution) + mutation
        neighbor = np.clip(neighbor, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        return neighbor

    def optimize(self):
        # Initialize a random solution
        current_solution = np.array([random.uniform(b[0], b[1]) for b in self.bounds])
        current_value = self.obj_func(current_solution)
        best_solution = current_solution.copy()
        best_value = current_value

        history = [(best_value, best_solution.tolist())]

        for iteration in range(self.max_iterations):
            # Select a random neighborhood size
            neighborhood_index = random.randint(0, self.num_neighborhoods - 1)
            neighborhood_size = self.neighborhood_sizes[neighborhood_index]

            # Generate neighbor
            neighbor = self._generate_neighbor(current_solution, neighborhood_size)
            neighbor_value = self.obj_func(neighbor)

            if neighbor_value < current_value:
                current_solution = neighbor
                current_value = neighbor_value

                # 🔥 Update global best if necessary
                if current_value < best_value:
                    best_solution = current_solution.copy()
                    best_value = current_value

            history.append((best_value, best_solution.tolist()))

        return best_solution.tolist(), best_value, history
