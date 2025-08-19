import numpy as np
import random

class TabuSearch:
    def __init__(self, objective_function, dim, bounds, max_iter=200, tabu_tenure=10, neighborhood_size=20):
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = bounds
        self.max_iter = max_iter
        self.tabu_tenure = tabu_tenure
        self.neighborhood_size = neighborhood_size

    def _generate_neighbor(self, current_solution):
        neighbor = np.array(current_solution) + np.random.uniform(-0.1, 0.1, self.dim)
        neighbor = np.clip(neighbor, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        return neighbor

    def _evaluate_neighbors(self, current_solution, tabu_list):
        best_candidate = None
        best_candidate_value = float('inf')
        candidate_move = None

        for _ in range(self.neighborhood_size):
            neighbor = self._generate_neighbor(current_solution)
            move = tuple(np.round(neighbor - current_solution, 4))

            if move in tabu_list:
                continue

            value = self.obj_func(neighbor)
            if value < best_candidate_value:
                best_candidate = neighbor
                best_candidate_value = value
                candidate_move = move

        return best_candidate, best_candidate_value, candidate_move

    def optimize(self):
        current_solution = np.array([random.uniform(b[0], b[1]) for b in self.bounds])
        current_value = self.obj_func(current_solution)

        best_solution = current_solution.copy()
        best_value = current_value

        tabu_list = {}
        history = [(current_value, current_solution.tolist())]

        for iteration in range(self.max_iter):
            neighbor, neighbor_value, move = self._evaluate_neighbors(current_solution, tabu_list)

            if neighbor is None:
                break

            current_solution = neighbor
            current_value = neighbor_value
            history.append((current_value, current_solution.tolist()))

            if current_value < best_value:
                best_solution = current_solution.copy()
                best_value = current_value

            # Update tabu list
            if move is not None:
                tabu_list[move] = self.tabu_tenure
            tabu_list = {k: v - 1 for k, v in tabu_list.items() if v > 1}

        return best_solution.tolist(), best_value, history

