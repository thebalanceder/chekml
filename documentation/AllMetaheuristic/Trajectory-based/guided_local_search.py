import numpy as np

class GuidedLocalSearch:
    def __init__(self, objective_function, feature_costs, feature_indicators,
                 dim=2, bounds=None, lambda_=0.1, max_iterations=100):
        """
        Initialize Guided Local Search (GLS).

        Parameters:
        - objective_function: Original objective function g(s) to minimize.
        - feature_costs: List of costs c_i for each feature.
        - feature_indicators: Function l_i(s) that returns list of binary indicators [l1, l2, ..., lM].
        - dim: Number of decision variables.
        - bounds: List of (lower, upper) tuples for each dimension.
        - lambda_: Penalty scaling factor λ.
        - max_iterations: Maximum number of iterations.
        """
        self.objective_function = objective_function
        self.feature_costs = np.array(feature_costs)
        self.feature_indicators_func = feature_indicators
        self.lambda_ = lambda_
        self.dim = dim
        self.bounds = bounds if bounds else [(-5, 5)] * dim
        self.max_iterations = max_iterations

        self.lower_bounds = np.array([b[0] for b in self.bounds])
        self.upper_bounds = np.array([b[1] for b in self.bounds])

        self.penalties = np.zeros(len(self.feature_costs))
        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    def _construct_initial_solution(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds)

    def _evaluate_augmented_objective(self, solution):
        g = self.objective_function(solution)
        l = np.array(self.feature_indicators_func(solution))
        penalty_term = np.dot(self.penalties, l)
        return g + self.lambda_ * penalty_term

    def _improvement_method(self, solution):
        # Naive local search by small random perturbation
        neighbor = solution + np.random.uniform(-0.1, 0.1, self.dim)
        neighbor = np.clip(neighbor, self.lower_bounds, self.upper_bounds)
        if self._evaluate_augmented_objective(neighbor) < self._evaluate_augmented_objective(solution):
            return neighbor
        return solution

    def optimize(self):
        k = 0
        s = self._construct_initial_solution()

        while k < self.max_iterations:
            s = self._improvement_method(s)

            l = np.array(self.feature_indicators_func(s))
            utilities = l * self.feature_costs / (1 + self.penalties)

            max_utility = np.max(utilities)
            max_indices = np.where(utilities == max_utility)[0]

            for i in max_indices:
                self.penalties[i] += 1

            current_fitness = self.objective_function(s)
            if current_fitness < self.best_fitness:
                self.best_fitness = current_fitness
                self.best_solution = s.copy()

            self.history.append((k, self.best_solution.copy()))
            print(f"Iteration {k + 1}: Best Fitness = {self.best_fitness}")
            k += 1

        return self.best_solution, self.best_fitness, self.history

