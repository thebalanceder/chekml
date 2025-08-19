import numpy as np

class StochasticFractalSearch:
    def __init__(self, objective_function, dim=2, bounds=None,
                 num_agents=30, max_iterations=100,
                 max_diffusion=4, walk_prob=0.8, show_result=True, plot=False):
        """
        Initialize the Stochastic Fractal Search (SFS) optimizer.

        Parameters:
        - objective_function: Function to be minimized.
        - dim: Number of decision variables.
        - bounds: List of (lower, upper) tuples for each dimension.
        - num_agents: Number of search agents.
        - max_iterations: Maximum number of iterations.
        - max_diffusion: Maximum number of diffusion steps.
        - walk_prob: Probability to select global walk over local walk.
        - show_result: Print best fitness each iteration.
        - plot: Not implemented, for future expansion.
        """
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = bounds if bounds else [(-5, 5)] * dim
        self.num_agents = num_agents
        self.max_iterations = max_iterations
        self.max_diffusion = max_diffusion
        self.walk_prob = walk_prob
        self.show_result = show_result
        self.plot = plot

        self.lower_bounds = np.array([b[0] for b in self.bounds])
        self.upper_bounds = np.array([b[1] for b in self.bounds])

        self.population = np.random.uniform(self.lower_bounds, self.upper_bounds,
                                            (self.num_agents, self.dim))
        self.fitness = np.array([self.obj_func(ind) for ind in self.population])
        self.history = []

    def _bound_checking(self, x):
        return np.clip(x, self.lower_bounds, self.upper_bounds)

    def _diffusion_process(self, point, generation, best_point):
        new_points = [point]
        for _ in range(self.max_diffusion):
            if np.random.rand() < self.walk_prob:
                sigma = (np.log(generation + 1) / (generation + 1)) * np.abs(point - best_point)
                generate_point = np.random.normal(best_point, sigma) + \
                                 (np.random.randn() * best_point - np.random.randn() * point)
            else:
                sigma = (np.log(generation + 1) / (generation + 1)) * np.abs(point - best_point)
                generate_point = np.random.normal(point, sigma)

            new_points.append(generate_point)

        new_points = self._bound_checking(np.array(new_points))
        fitness = np.array([self.obj_func(p) for p in new_points])
        best_idx = np.argmin(fitness)
        return new_points[best_idx], fitness[best_idx]

    def optimize(self):
        idx_sorted = np.argsort(self.fitness)
        self.population = self.population[idx_sorted]
        self.fitness = self.fitness[idx_sorted]
        best_point = self.population[0].copy()
        best_fitness = self.fitness[0]
        fitness_history = [best_fitness]

        for generation in range(self.max_iterations):
            new_population = []
            new_fitness = []

            for i in range(self.num_agents):
                new_point, new_fit = self._diffusion_process(self.population[i], generation, best_point)
                new_population.append(new_point)
                new_fitness.append(new_fit)

            new_population = np.array(new_population)
            new_fitness = np.array(new_fitness)

            # First Update Process
            sorted_idx = np.argsort(new_fitness)
            Pa = (self.num_agents - np.arange(1, self.num_agents + 1) + 1) / self.num_agents
            rand1 = np.random.permutation(self.num_agents)
            rand2 = np.random.permutation(self.num_agents)
            P = new_population.copy()

            for i in range(self.num_agents):
                for j in range(self.dim):
                    if np.random.rand() > Pa[i]:
                        P[i, j] = new_population[rand1[i], j] - \
                                  np.random.rand() * (new_population[rand2[i], j] - new_population[i, j])

            P = self._bound_checking(P)
            fit_first_process = np.array([self.obj_func(p) for p in P])

            for i in range(self.num_agents):
                if fit_first_process[i] <= new_fitness[i]:
                    new_population[i] = P[i]
                    new_fitness[i] = fit_first_process[i]

            # Sort and keep best
            sorted_idx = np.argsort(new_fitness)
            new_population = new_population[sorted_idx]
            new_fitness = new_fitness[sorted_idx]
            if new_fitness[0] < best_fitness:
                best_fitness = new_fitness[0]
                best_point = new_population[0].copy()

            fitness_history.append(best_fitness)

            # Second Update Process
            Pa = np.sort(sorted_idx / self.num_agents)[::-1]
            for i in range(self.num_agents):
                if np.random.rand() > Pa[i]:
                    r1, r2 = np.random.randint(0, self.num_agents, 2)
                    while r1 == r2:
                        r2 = np.random.randint(0, self.num_agents)
                    if np.random.rand() < 0.5:
                        replaced = new_population[i] - np.random.rand() * (new_population[r2] - best_point)
                    else:
                        replaced = new_population[i] + np.random.rand() * (new_population[r2] - new_population[r1])
                    replaced = self._bound_checking(replaced)
                    if self.obj_func(replaced) < self.obj_func(new_population[i]):
                        new_population[i] = replaced

            self.population = new_population
            self.fitness = new_fitness

            if self.show_result:
                print(f"Iteration {generation + 1}: Best Fitness = {best_fitness:.6e}")

            self.history.append((generation, best_point.copy()))

        return best_point, best_fitness, self.history

