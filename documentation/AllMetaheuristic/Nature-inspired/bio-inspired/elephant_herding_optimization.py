import numpy as np

class ElephantHerdingOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_fes=30000, 
                 num_clans=5, alpha=0.5, beta=0.1, keep=2):
        """
        Initialize the Elephant Herding Optimization algorithm.

        Parameters:
        - objective_function: Function to optimize (returns cost, assumes lower is better).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Total number of elephants (solutions).
        - max_fes: Maximum number of function evaluations.
        - num_clans: Number of clans to divide the population into.
        - alpha: Clan updating factor for movement towards best elephant.
        - beta: Clan center influence factor for unchanged elephants.
        - keep: Number of elite elephants to preserve (elitism).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_fes = max_fes
        self.num_clans = num_clans
        self.alpha = alpha
        self.beta = beta
        self.keep = keep
        self.num_elephants_per_clan = population_size // num_clans
        if population_size % num_clans != 0:
            raise ValueError("Population size must be divisible by number of clans")
        self.population = None
        self.costs = None
        self.best_solution = None
        self.best_cost = float("inf")
        self.n_evaluations = 0
        self.history = []

    def initialize_population(self):
        """ Generate initial population randomly within bounds """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                            (self.population_size, self.dim))
        self.costs = np.array([self.objective_function(ind) for ind in self.population])
        self.n_evaluations += self.population_size
        self._update_best()
        self._clear_duplicates()

    def _update_best(self):
        """ Update the best solution and cost """
        min_idx = np.argmin(self.costs)
        if self.costs[min_idx] < self.best_cost:
            self.best_solution = self.population[min_idx].copy()
            self.best_cost = self.costs[min_idx]

    def _clear_duplicates(self):
        """ Ensure no duplicate individuals in the population """
        for i in range(self.population_size):
            chrom1 = np.sort(self.population[i])
            for j in range(i + 1, self.population_size):
                chrom2 = np.sort(self.population[j])
                if np.array_equal(chrom1, chrom2):
                    parnum = np.random.randint(0, self.dim)
                    self.population[j, parnum] = np.random.uniform(
                        self.bounds[parnum, 0], self.bounds[parnum, 1])

    def _sort_population(self):
        """ Sort population by cost (ascending) """
        indices = np.argsort(self.costs)
        self.population = self.population[indices]
        self.costs = self.costs[indices]
        return indices

    def _compute_clan_center(self, clan):
        """ Calculate the center of a clan """
        return np.mean(clan, axis=0)

    def _divide_into_clans(self):
        """ Divide population into clans based on fitness """
        self._sort_population()  # Ensure population is sorted by fitness
        clans = [[] for _ in range(self.num_clans)]
        popindex = 0
        j = 0
        while popindex < self.population_size:
            for cindex in range(self.num_clans):
                clans[cindex].append(self.population[popindex])
                popindex += 1
            j += 1
        return [np.array(clan) for clan in clans]

    def _clan_updating(self, clans):
        """ Update clans using clan updating operator """
        new_clans = [[] for _ in range(self.num_clans)]
        popindex = 0
        j = 0
        while popindex < self.population_size:
            for cindex in range(self.num_clans):
                clan_center = self._compute_clan_center(clans[cindex])
                best_elephant = clans[cindex][0]  # Best in clan (after sorting)
                current = clans[cindex][j]
                new_chrom = current + self.alpha * (best_elephant - current) * np.random.rand(self.dim)
                if np.sum(new_chrom - current) == 0:
                    new_chrom = self.beta * clan_center
                new_clans[cindex].append(np.clip(new_chrom, self.bounds[:, 0], self.bounds[:, 1]))
                popindex += 1
            j += 1
        return [np.array(clan) for clan in new_clans]

    def _separating_operator(self, new_clans):
        """ Apply separating operator to the worst elephant in each clan """
        for cindex in range(self.num_clans):
            new_clans[cindex][-1] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)

    def _evaluate_clans(self, new_clans):
        """ Evaluate new clans and ensure feasibility """
        for i in range(self.num_clans):
            clan = np.array(new_clans[i])
            # Clear duplicates
            for j in range(len(clan)):
                chrom1 = np.sort(clan[j])
                for k in range(j + 1, len(clan)):
                    chrom2 = np.sort(clan[k])
                    if np.array_equal(chrom1, chrom2):
                        parnum = np.random.randint(0, self.dim)
                        clan[k, parnum] = np.random.uniform(
                            self.bounds[parnum, 0], self.bounds[parnum, 1])
            # Evaluate costs
            costs = np.array([self.objective_function(ind) for ind in clan])
            self.n_evaluations += len(clan)
            # Sort clan
            indices = np.argsort(costs)
            new_clans[i] = clan[indices]
        return new_clans

    def _combine_clans(self, new_clans):
        """ Combine clans into a single population """
        population = []
        j = 0
        popindex = 0
        while popindex < self.population_size and j < self.num_elephants_per_clan:
            for cindex in range(self.num_clans):
                population.append(new_clans[cindex][j])
                popindex += 1
            j += 1
        return np.array(population)

    def _apply_elitism(self):
        """ Replace worst individuals with elite solutions """
        elite_chroms = self.population[:self.keep].copy()
        elite_costs = self.costs[:self.keep].copy()
        self._sort_population()
        self.population[-self.keep:] = elite_chroms
        self.costs[-self.keep:] = elite_costs

    def optimize(self):
        """ Run the Elephant Herding Optimization algorithm """
        self.initialize_population()
        gen_index = 1
        while self.n_evaluations < self.max_fes:
            # Save elites
            elite_chroms = self.population[:self.keep].copy()
            elite_costs = self.costs[:self.keep].copy()

            # Divide into clans
            clans = self._divide_into_clans()

            # Clan updating
            new_clans = self._clan_updating(clans)

            # Separating operator
            self._separating_operator(new_clans)

            # Evaluate clans
            new_clans = self._evaluate_clans(new_clans)

            # Combine clans
            self.population = self._combine_clans(new_clans)
            self.costs = np.array([self.objective_function(ind) for ind in self.population])
            self.n_evaluations += self.population_size

            # Apply elitism
            self.population[-self.keep:] = elite_chroms
            self.costs[-self.keep:] = elite_costs

            # Sort and update best
            self._sort_population()
            self._update_best()

            # Compute average cost
            legal_costs = self.costs[self.costs < float("inf")]
            avg_cost = np.mean(legal_costs) if len(legal_costs) > 0 else float("inf")

            # Log results
            self.history.append((gen_index, self.best_solution.copy(), self.best_cost, avg_cost))
            print(f"Generation {gen_index}: Best Cost = {self.best_cost}, Avg Cost = {avg_cost}")

            gen_index += 1

        return self.best_solution, self.best_cost, self.history
