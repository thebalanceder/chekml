import numpy as np

class FrogLeapingOptimizer:
    def __init__(self, objective_function=None, dim=10, bounds=(-10, 10), 
                 max_iter=1000, memeplex_size=10, num_memeplexes=5, 
                 num_parents=None, num_offsprings=3, max_fla_iter=5, step_size=2):
        """
        Initialize the Shuffled Frog Leaping Algorithm (SFLA) optimizer.

        Parameters:
        - objective_function: Function to optimize (default: Sphere function).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - max_iter: Maximum number of iterations.
        - memeplex_size: Number of frogs per memeplex.
        - num_memeplexes: Number of memeplexes.
        - num_parents: Number of parents for FLA (default: 0.3 * memeplex_size).
        - num_offsprings: Number of offsprings per FLA iteration.
        - max_fla_iter: Maximum iterations for FLA.
        - step_size: Step size for position updates.
        """
        self.objective_function = objective_function 
        self.dim = dim
        self.bounds = np.array([bounds] * dim)  # Uniform bounds for each dimension
        self.max_iter = max_iter
        self.memeplex_size = max(memeplex_size, dim + 1)  # Ensure Nelder-Mead standard
        self.num_memeplexes = num_memeplexes
        self.population_size = self.memeplex_size * num_memeplexes
        self.num_parents = num_parents if num_parents is not None else max(round(0.3 * self.memeplex_size), 2)
        self.num_offsprings = num_offsprings
        self.max_fla_iter = max_fla_iter
        self.step_size = step_size

        self.population = None  # Population of frogs (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

        # Memeplex indices
        self.memeplex_indices = np.reshape(np.arange(self.population_size), (num_memeplexes, self.memeplex_size))

    def initialize_population(self):
        """Generate initial population of frogs randomly."""
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim))

    def evaluate_population(self):
        """Compute fitness values for the population."""
        return np.array([self.objective_function(frog) for frog in self.population])

    def sort_population(self, population=None, fitness=None):
        """Sort population by fitness."""
        if population is None:
            population = self.population
            fitness = self.evaluate_population()
        sort_order = np.argsort(fitness)
        return population[sort_order], sort_order

    def is_in_range(self, x):
        """Check if position is within bounds."""
        return np.all(x >= self.bounds[:, 0]) and np.all(x <= self.bounds[:, 1])

    def rand_sample(self, probabilities, num_samples, replacement=False):
        """Random sampling with given probabilities."""
        choices = np.arange(len(probabilities))
        samples = []
        probs = probabilities.copy()
        for _ in range(num_samples):
            idx = np.random.choice(choices, p=probs / probs.sum())
            samples.append(idx)
            if not replacement:
                probs[idx] = 0
        return np.array(samples)

    def run_fla(self, memeplex):
        """Run Frog Leaping Algorithm on a memeplex."""
        n_pop = len(memeplex)
        # Selection probabilities (triangular distribution)
        P = 2 * (n_pop + 1 - np.arange(1, n_pop + 1)) / (n_pop * (n_pop + 1))
        
        # Calculate memeplex range (smallest hypercube)
        lower_bound = memeplex[0].copy()
        upper_bound = memeplex[0].copy()
        for frog in memeplex[1:]:
            lower_bound = np.minimum(lower_bound, frog)
            upper_bound = np.maximum(upper_bound, frog)

        # FLA main loop
        for _ in range(self.max_fla_iter):
            # Select parents
            parent_indices = self.rand_sample(P, self.num_parents)
            subcomplex = memeplex[parent_indices]

            # Generate offsprings
            for _ in range(self.num_offsprings):
                # Sort subcomplex
                sub_fitness = np.array([self.objective_function(frog) for frog in subcomplex])
                sorted_indices = np.argsort(sub_fitness)
                subcomplex = subcomplex[sorted_indices]
                parent_indices = parent_indices[sorted_indices]

                # Improvement Step 1: Move worst towards best in subcomplex
                new_solution = subcomplex[-1].copy()
                step = self.step_size * np.random.rand(self.dim) * (subcomplex[0] - subcomplex[-1])
                new_solution += step
                improvement_step2 = False
                censorship = False

                if self.is_in_range(new_solution):
                    new_cost = self.objective_function(new_solution)
                    if new_cost < sub_fitness[-1]:
                        subcomplex[-1] = new_solution
                        sub_fitness[-1] = new_cost
                    else:
                        improvement_step2 = True
                else:
                    improvement_step2 = True

                # Improvement Step 2: Move worst towards global best
                if improvement_step2:
                    new_solution = subcomplex[-1].copy()
                    step = self.step_size * np.random.rand(self.dim) * (self.best_solution - subcomplex[-1])
                    new_solution += step
                    if self.is_in_range(new_solution):
                        new_cost = self.objective_function(new_solution)
                        if new_cost < sub_fitness[-1]:
                            subcomplex[-1] = new_solution
                            sub_fitness[-1] = new_cost
                        else:
                            censorship = True
                    else:
                        censorship = True

                # Censorship: Replace worst with random position
                if censorship:
                    subcomplex[-1] = np.random.uniform(lower_bound, upper_bound, self.dim)
                    sub_fitness[-1] = self.objective_function(subcomplex[-1])

                # Update memeplex
                memeplex[parent_indices] = subcomplex

        return memeplex

    def optimize(self):
        """Run the Shuffled Frog Leaping Algorithm."""
        self.initialize_population()
        self.population, _ = self.sort_population()
        self.best_solution = self.population[0].copy()
        self.best_value = self.objective_function(self.best_solution)

        for iteration in range(self.max_iter):
            memeplexes = []
            # Form and process memeplexes
            for j in range(self.num_memeplexes):
                memeplex = self.population[self.memeplex_indices[j]].copy()
                memeplex = self.run_fla(memeplex)
                memeplexes.append(memeplex)
                self.population[self.memeplex_indices[j]] = memeplex

            # Sort population
            self.population, _ = self.sort_population()
            self.best_solution = self.population[0].copy()
            self.best_value = self.objective_function(self.best_solution)

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

