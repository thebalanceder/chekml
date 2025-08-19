import numpy as np


class IterativeLocalSearch:
    def __init__(self, objective_function, bounds, n_iterations=1000, step_size=0.05,
                 n_restarts=30, perturbation_size=1.0, seed_value=1):
        """
        Initialize Iterated Local Search (ILS).

        Parameters:
        - objective_function: Objective function to minimize.
        - bounds: Array of (lower, upper) tuples for each dimension.
        - n_iterations: Number of local search steps per restart.
        - step_size: Step size for local search.
        - n_restarts: Number of random restarts.
        - perturbation_size: Magnitude of perturbation for generating new initial points.
        - seed_value: Random seed for reproducibility.
        """
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.n_restarts = n_restarts
        self.perturbation_size = perturbation_size
        self.dim = len(bounds)
        self.rng = np.random.default_rng(seed_value)

        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    def _in_bounds(self, point):
        return np.all(point >= self.bounds[:, 0]) and np.all(point <= self.bounds[:, 1])

    def _random_point_within_bounds(self):
        return self.bounds[:, 0] + self.rng.random(self.dim) * (self.bounds[:, 1] - self.bounds[:, 0])

    def _hill_climb(self, start_point):
        solution = start_point
        solution_eval = self.objective_function(solution)

        for _ in range(self.n_iterations):
            candidate = None
            while candidate is None or not self._in_bounds(candidate):
                candidate = solution + self.rng.normal(0, self.step_size, self.dim)

            candidate_eval = self.objective_function(candidate)

            if candidate_eval <= solution_eval:
                solution, solution_eval = candidate, candidate_eval

        return solution, solution_eval

    def optimize(self):
        self.best_solution = self._random_point_within_bounds()
        self.best_fitness = self.objective_function(self.best_solution)

        for restart in range(self.n_restarts):
            start_pt = None
            while start_pt is None or not self._in_bounds(start_pt):
                start_pt = self.best_solution + self.rng.normal(0, self.perturbation_size, self.dim)

            solution, solution_eval = self._hill_climb(start_pt)

            if solution_eval < self.best_fitness:
                self.best_solution = solution
                self.best_fitness = solution_eval
                print(f"Restart {restart}: Best Fitness = {self.best_fitness:.5f}")

            self.history.append((restart, self.best_solution.copy(), self.best_fitness))

        return self.best_solution, self.best_fitness, self.history
