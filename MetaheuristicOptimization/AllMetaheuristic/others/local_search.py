import numpy as np

class LocalSearch:
    def __init__(self, objective_function, dim, bounds, max_iter=100, step_size=0.1, neighbor_count=10):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.max_iter = max_iter
        self.step_size = step_size  # Controls the magnitude of local search steps
        self.neighbor_count = neighbor_count  # Number of neighbors per iteration

        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def random_solution(self):
        """ Generate a random solution within the given bounds """
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)

    def generate_neighbors(self, current_solution):
        """ Generate neighboring solutions by adding small perturbations """
        neighbors = []
        for _ in range(self.neighbor_count):
            perturbation = np.random.uniform(-self.step_size, self.step_size, self.dim)
            new_solution = current_solution + perturbation
            new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])  # Keep within bounds
            neighbors.append(new_solution)
        return neighbors

    def optimize(self):
        """ Run the Local Search algorithm """
        current_solution = self.random_solution()
        current_value = self.objective_function(current_solution)

        self.best_solution = current_solution
        self.best_value = current_value

        for iteration in range(self.max_iter):
            neighbors = self.generate_neighbors(current_solution)

            # Evaluate all neighbors and select the best
            best_neighbor = current_solution
            best_neighbor_value = current_value

            for neighbor in neighbors:
                neighbor_value = self.objective_function(neighbor)
                if neighbor_value < best_neighbor_value:
                    best_neighbor = neighbor
                    best_neighbor_value = neighbor_value

            # If no better neighbor found, stop (local optimum reached)
            if best_neighbor_value >= current_value:
                print(f"Stopping at iteration {iteration + 1}: No better neighbor found.")
                break

            # Move to the best neighbor
            current_solution = best_neighbor
            current_value = best_neighbor_value

            # Store search history for visualization
            self.history.append((iteration, current_solution.copy(), current_value))

            # Update global best
            if best_neighbor_value < self.best_value:
                self.best_solution = best_neighbor
                self.best_value = best_neighbor_value

            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

