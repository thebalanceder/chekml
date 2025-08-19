import numpy as np

class FireHawkOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=25, max_fes=100):
        """
        Initialize the Fire Hawk Optimizer (FHO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of initial candidates (Fire Hawks and Prey).
        - max_fes: Maximum number of function evaluations.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_fes = max_fes
        self.num_firehawks = np.random.randint(1, int(np.ceil(population_size / 5)) + 1)

        self.population = None  # Population of solutions
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []
        self.fes = 0  # Function evaluations counter
        self.iteration = 0

    def initialize_population(self):
        """ Generate initial population randomly """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim))
        costs = self.evaluate_population()
        sort_order = np.argsort(costs)
        self.population = self.population[sort_order]
        costs = costs[sort_order]
        self.best_solution = self.population[0].copy()
        self.best_value = costs[0]

    def evaluate_population(self):
        """ Compute fitness values for the population """
        costs = np.array([self.objective_function(ind) for ind in self.population])
        self.fes += len(self.population)
        return costs

    def distance(self, a, b):
        """ Calculate Euclidean distance between two points """
        return np.sqrt(np.sum((a - b) ** 2))

    def assign_prey_to_firehawks(self, firehawks, prey):
        """ Assign prey to Fire Hawks based on distance """
        prey_groups = []
        prey_remaining = prey.copy()
        for fh in firehawks:
            if len(prey_remaining) == 0:
                break
            distances = np.array([self.distance(fh, p) for p in prey_remaining])
            sort_indices = np.argsort(distances)
            num_prey = np.random.randint(1, len(prey_remaining) + 1)
            selected_indices = sort_indices[:num_prey]
            prey_groups.append(prey_remaining[selected_indices])
            prey_remaining = np.delete(prey_remaining, selected_indices, axis=0)
        if len(prey_remaining) > 0:
            if prey_groups:
                prey_groups[-1] = np.vstack((prey_groups[-1], prey_remaining))
            else:
                prey_groups.append(prey_remaining)
        return prey_groups

    def update_firehawk_position(self, fh, other_fh):
        """ Update Fire Hawk position """
        ir = np.random.uniform(0, 1, 2)
        new_pos = fh + (ir[0] * self.best_solution - ir[1] * other_fh)
        return np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])

    def update_prey_position(self, prey, firehawk, safe_point, global_safe_point):
        """ Update Prey position with two strategies """
        ir1 = np.random.uniform(0, 1, 2)
        pos1 = prey + (ir1[0] * firehawk - ir1[1] * safe_point)
        pos1 = np.clip(pos1, self.bounds[:, 0], self.bounds[:, 1])

        ir2 = np.random.uniform(0, 1, 2)
        other_fh = self.population[np.random.randint(0, self.num_firehawks)]
        pos2 = prey + (ir2[0] * other_fh - ir2[1] * global_safe_point)
        pos2 = np.clip(pos2, self.bounds[:, 0], self.bounds[:, 1])

        return pos1, pos2

    def optimize(self):
        """ Run the Fire Hawk Optimization algorithm """
        self.initialize_population()
        global_safe_point = np.mean(self.population, axis=0)

        while self.fes < self.max_fes:
            self.iteration += 1
            self.num_firehawks = np.random.randint(1, int(np.ceil(self.population_size / 5)) + 1)
            firehawks = self.population[:self.num_firehawks]
            prey = self.population[self.num_firehawks:] if self.num_firehawks < self.population_size else np.array([])

            # Assign prey to Fire Hawks
            prey_groups = self.assign_prey_to_firehawks(firehawks, prey) if len(prey) > 0 else []

            new_population = []
            for i, fh in enumerate(firehawks):
                # Update Fire Hawk
                other_fh = firehawks[np.random.randint(0, len(firehawks))]
                new_fh = self.update_firehawk_position(fh, other_fh)
                new_population.append(new_fh)

                # Update Prey
                if i < len(prey_groups) and len(prey_groups[i]) > 0:
                    local_safe_point = np.mean(prey_groups[i], axis=0)
                    for p in prey_groups[i]:
                        pos1, pos2 = self.update_prey_position(p, fh, local_safe_point, global_safe_point)
                        new_population.extend([pos1, pos2])

            # Evaluate new population
            new_population = np.array(new_population)
            if len(new_population) > 0:
                costs = np.array([self.objective_function(ind) for ind in new_population])
                self.fes += len(new_population)
                sort_order = np.argsort(costs)
                new_population = new_population[sort_order]
                costs = costs[sort_order]
                self.population = new_population[:self.population_size] if len(new_population) > self.population_size else new_population
                if costs[0] < self.best_value:
                    self.best_solution = self.population[0].copy()
                    self.best_value = costs[0]

            global_safe_point = np.mean(self.population, axis=0)
            self.history.append((self.iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {self.iteration}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

