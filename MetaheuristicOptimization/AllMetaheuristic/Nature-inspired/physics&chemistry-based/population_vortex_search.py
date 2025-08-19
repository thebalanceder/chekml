import numpy as np

class PopulationVortexSearch:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=1000, prob_mut=None, prob_cross=None):
        """
        Initialize the Population-based Vortex Search (PVS) optimizer.
        
        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of candidate solutions.
        - max_iter: Maximum number of iterations (function evaluations).
        - prob_mut: Probability of mutation (default: 1/dim).
        - prob_cross: Probability of crossover (default: 1/dim).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.prob_mut = prob_mut if prob_mut is not None else 1/dim
        self.prob_cross = prob_cross if prob_cross is not None else 1/dim
        self.vortex_size = population_size // 2
        self.distribution_index = 20

        self.center = None  # Center of the vortex (Mu)
        self.best_value = float("inf")  # Global minimum fitness (gmin)
        self.best_solution = None  # Best solution found (Mu)
        self.candidates = None  # Candidate solutions (Cs)
        self.iter_results = []  # Store iteration results
        self.function_evals = 0  # Function evaluations counter

    def initialize_vortex(self):
        """ Initialize the vortex center and candidate solutions """
        LB, UB = self.bounds[:, 0], self.bounds[:, 1]
        self.center = 0.5 * (UB + LB)  # Initial center (Mu)
        x = 0.1  # For gammaincinv
        a = 1
        ginv = (1/x) * np.reciprocal(np.random.gamma(a, scale=x))  # Approximate gammaincinv
        radius = ginv * ((UB - LB) / 2)  # Initial radius
        self.candidates = np.random.normal(loc=self.center, scale=radius, size=(self.population_size, self.dim))
        self._bound_solutions()

    def _bound_solutions(self):
        """ Ensure candidate solutions are within bounds """
        LB, UB = self.bounds[:, 0], self.bounds[:, 1]
        for i in range(self.population_size):
            below = self.candidates[i] < LB
            above = self.candidates[i] > UB
            self.candidates[i][below] = np.random.uniform(LB[below], UB[below])
            self.candidates[i][above] = np.random.uniform(LB[above], UB[above])

    def evaluate_candidates(self, indices=None):
        """ Evaluate the objective function for candidate solutions """
        if indices is None:
            return np.array([self.objective_function(sol) for sol in self.candidates])
        return np.array([self.objective_function(self.candidates[i]) for i in indices])

    def polynomial_mutation(self, solution):
        """ Apply polynomial mutation to a solution """
        LB, UB = self.bounds[:, 0], self.bounds[:, 1]
        mutated = solution.copy()
        state = 0
        mut_pow = 1 / (1 + self.distribution_index)

        for i in range(self.dim):
            if np.random.rand() <= self.prob_mut:
                y = solution[i]
                yL, yU = LB[i], UB[i]
                delta1 = (y - yL) / (yU - yL)
                delta2 = (yU - y) / (yU - yL)
                rnd = np.random.rand()

                if rnd <= 0.5:
                    xy = 1 - delta1
                    val = 2 * rnd + (1 - 2 * rnd) * xy ** (self.distribution_index + 1)
                    deltaq = val ** mut_pow - 1
                else:
                    xy = 1 - delta2
                    val = 2 * (1 - rnd) + 2 * (rnd - 0.5) * xy ** (self.distribution_index + 1)
                    deltaq = 1 - val ** mut_pow

                y = y + deltaq * (yU - yL)
                mutated[i] = np.clip(y, yL, yU)
                state += 1

        return mutated, state

    def first_phase(self, radius, iteration):
        """ Generate new candidate solutions around the vortex center """
        if iteration == 0:
            candidates = np.random.normal(loc=self.center, scale=radius, size=(self.population_size, self.dim))
            self.candidates = candidates
            self._bound_solutions()
            return self.evaluate_candidates()
        else:
            candidates = np.random.normal(loc=self.center, scale=radius, size=(self.vortex_size, self.dim))
            self.candidates[:self.vortex_size] = candidates
            self._bound_solutions()
            return self.evaluate_candidates(range(self.vortex_size))

    def second_phase(self, obj_vals, iteration):
        """ Apply crossover and mutation to remaining candidates """
        LB, UB = self.bounds[:, 0], self.bounds[:, 1]
        # Ensure obj_vals is full size for all candidates
        if iteration > 0:
            full_obj_vals = self.evaluate_candidates()
            full_obj_vals[:self.vortex_size] = obj_vals
            obj_vals = full_obj_vals
            self.function_evals += self.population_size - self.vortex_size
        else:
            obj_vals = obj_vals.copy()

        prob = 0.9 * (np.max(obj_vals) - obj_vals) + 0.1
        prob = np.cumsum(prob / np.sum(prob))
        
        for i in range(self.vortex_size, self.population_size):
            neighbor = np.searchsorted(prob, np.random.rand())
            while i == neighbor:
                neighbor = np.searchsorted(prob, np.random.rand())

            sol = self.candidates[i].copy()
            param2change = np.random.randint(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.prob_cross or d == param2change:
                    sol[d] += (self.candidates[i, d] - self.candidates[neighbor, d]) * (np.random.rand() - 0.5) * 2

            sol = np.clip(sol, LB, UB)
            obj_val_sol = self.objective_function(sol)
            self.function_evals += 1

            if obj_val_sol < obj_vals[i]:
                self.candidates[i] = sol
                obj_vals[i] = obj_val_sol
            else:
                mutated, state = self.polynomial_mutation(self.candidates[i])
                if state > 0:
                    obj_val_mut = self.objective_function(mutated)
                    self.function_evals += 1
                    if obj_val_mut < obj_vals[i]:
                        self.candidates[i] = mutated
                        obj_vals[i] = obj_val_mut

        return obj_vals

    def optimize(self, error=1e-6, optimal_value=0):
        """ Run the Population-based Vortex Search optimization """
        self.initialize_vortex()
        LB, UB = self.bounds[:, 0], self.bounds[:, 1]
        x = 0.1
        iteration = 0

        while self.function_evals < self.max_iter:
            a = (self.max_iter - self.function_evals) / self.max_iter
            a = max(a, 0.1)
            ginv = (1/x) * np.reciprocal(np.random.gamma(a, scale=x))  # Approximate gammaincinv
            radius = ginv * ((UB - LB) / 2)

            # First phase: Generate and evaluate candidates
            obj_vals = self.first_phase(radius, iteration)
            self.function_evals += self.population_size if iteration == 0 else self.vortex_size

            # Update center
            min_idx = np.argmin(obj_vals)
            fmin = obj_vals[min_idx]
            itr_best = self.candidates[min_idx]

            if fmin < self.best_value:
                self.best_value = fmin
                self.best_solution = itr_best
                self.center = itr_best

            # Second phase: Crossover and mutation
            obj_vals = self.second_phase(obj_vals, iteration)

            # Update center again
            min_idx = np.argmin(obj_vals)
            fmin = obj_vals[min_idx]
            itr_best = self.candidates[min_idx]

            if fmin < self.best_value:
                self.best_value = fmin
                self.best_solution = itr_best
                self.center = itr_best

            # Store iteration results
            self.iter_results.append({"iteration": iteration, "best_value": self.best_value, "best_solution": self.best_solution.copy()})
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

            # Check termination condition
            if abs(optimal_value - self.best_value) <= error:
                break

            iteration += 1

        return self.best_solution, self.best_value, self.iter_results
