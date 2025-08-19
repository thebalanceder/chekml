import numpy as np

class PartialReinforcementOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=30, max_evaluations=100, 
                 reinforcement_rate=0.7):
        """
        Initialize the Partial Reinforcement Optimizer (PRO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of individuals (solutions).
        - max_evaluations: Maximum number of function evaluations.
        - reinforcement_rate: Reinforcement rate (RR) for schedule updates.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.reinforcement_rate = reinforcement_rate

        self.population = None  # Population of solutions
        self.schedules = None  # Schedules for behavior selection
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []
        self.evaluations = 0

    def initialize_population(self):
        """ Generate initial population and schedules randomly """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim))
        self.schedules = np.random.uniform(0.9, 1.0, (self.population_size, self.dim))

    def evaluate_population(self):
        """ Compute fitness values for the population """
        return np.array([self.objective_function(ind) for ind in self.population])

    def select_behaviors(self, i, current_eval):
        """ Select behaviors based on schedule priorities """
        tau = current_eval / self.max_evaluations  # Time parameter
        selection_rate = np.exp(-(1 - tau))  # Selection rate for behavior selection
        sorted_indices = np.argsort(self.schedules[i])[::-1]  # Sort in descending order
        landa = int(np.ceil(self.dim * np.random.rand() * selection_rate))
        return sorted_indices[:landa]

    def stimulate_behaviors(self, i, selected_behaviors, k, current_eval):
        """ Stimulate selected behaviors to generate new solution """
        tau = current_eval / self.max_evaluations
        stimulation = np.zeros(self.dim)

        if np.random.rand() < 0.5:
            stimulation[selected_behaviors] = (self.best_solution[selected_behaviors] - 
                                              self.population[i][selected_behaviors])
        else:
            stimulation[selected_behaviors] = (self.population[i][selected_behaviors] - 
                                              self.population[k][selected_behaviors])

        # Calculate Stimulation Factor (SF)
        sf = tau + np.random.rand() * (np.mean(self.schedules[i][selected_behaviors]) / 
                                       np.max(np.abs(self.schedules[i])))
        
        new_solution = self.population[i].copy()
        new_solution[selected_behaviors] += sf * stimulation[selected_behaviors]

        # Bound constraints control
        new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
        under_lb = new_solution < self.bounds[:, 0]
        over_ub = new_solution > self.bounds[:, 1]
        if np.any(under_lb):
            new_solution[under_lb] = self.bounds[:, 0][under_lb] + np.random.rand(np.sum(under_lb)) * \
                                     (self.bounds[:, 1][under_lb] - self.bounds[:, 0][under_lb])
        if np.any(over_ub):
            new_solution[over_ub] = self.bounds[:, 0][over_ub] + np.random.rand(np.sum(over_ub)) * \
                                    (self.bounds[:, 1][over_ub] - self.bounds[:, 0][over_ub])

        return new_solution

    def apply_reinforcement(self, i, selected_behaviors, new_solution, new_value):
        """ Apply positive or negative reinforcement based on response """
        current_value = self.objective_function(self.population[i])
        if new_value < current_value:
            # Positive Reinforcement
            self.schedules[i][selected_behaviors] += self.schedules[i][selected_behaviors] * (self.reinforcement_rate / 2)
            self.population[i] = new_solution
            if new_value < self.best_value:
                self.best_solution = new_solution.copy()
                self.best_value = new_value
        else:
            # Negative Reinforcement
            self.schedules[i][selected_behaviors] -= self.schedules[i][selected_behaviors] * self.reinforcement_rate

    def reschedule(self, i):
        """ Reschedule if schedule variance is zero """
        if np.std(self.schedules[i]) == 0:
            self.schedules[i] = np.random.uniform(0.9, 1.0, self.dim)
            self.population[i] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
            print(f"Learner {i} is Rescheduled")

    def optimize(self):
        """ Run the Partial Reinforcement Optimization """
        self.initialize_population()
        self.best_solution = self.population[0].copy()
        self.best_value = self.objective_function(self.best_solution)

        while self.evaluations < self.max_evaluations:
            fitness = self.evaluate_population()
            sorted_indices = np.argsort(fitness)
            self.population = self.population[sorted_indices]
            self.schedules = self.schedules[sorted_indices]
            if fitness[sorted_indices[0]] < self.best_value:
                self.best_solution = self.population[0].copy()
                self.best_value = fitness[sorted_indices[0]]

            for i in range(self.population_size):
                # Select another individual for comparison
                k = self.population_size - 1
                if i < self.population_size - 1:
                    k = sorted_indices[np.random.randint(i + 1, self.population_size)]

                # Select behaviors
                selected_behaviors = self.select_behaviors(i, self.evaluations)

                # Stimulate behaviors
                new_solution = self.stimulate_behaviors(i, selected_behaviors, k, self.evaluations)

                # Evaluate new solution
                new_value = self.objective_function(new_solution)
                self.evaluations += 1

                # Apply reinforcement
                self.apply_reinforcement(i, selected_behaviors, new_solution, new_value)

                # Reschedule if necessary
                self.reschedule(i)

                # Store best cost for current evaluation
                self.history.append((self.evaluations, self.best_solution.copy(), self.best_value))
                
                # Check if max evaluations reached
                if self.evaluations >= self.max_evaluations:
                    break

            print(f"Iteration {self.evaluations}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
