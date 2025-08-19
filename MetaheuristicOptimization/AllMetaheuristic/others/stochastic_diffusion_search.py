import numpy as np

class StochasticDiffusionSearch:
    def __init__(self, objective_function, dim, bounds, population_size=1000, max_iter=100,
                 mutation_rate=0.08, mutation_scale=4.0, cluster_threshold=0.33, 
                 context_sensitive=False):
        """
        Initialize the Stochastic Diffusion Search (SDS) optimizer.

        Parameters:
        - objective_function: Function to optimize, returns a list of component values.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of agents (hypotheses).
        - max_iter: Maximum number of iterations.
        - mutation_rate: Probability of applying mutation during diffusion (for dynamic problems).
        - mutation_scale: Controls the standard deviation of mutation offset.
        - cluster_threshold: Fraction of agents required in a cluster to consider convergence.
        - context_sensitive: If True, uses context-sensitive diffusion to balance exploration.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.cluster_threshold = cluster_threshold
        self.context_sensitive = context_sensitive

        self.agents = None  # Population of agent hypotheses (solutions)
        self.activities = None  # Boolean array indicating active/inactive agents
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []
        self.component_functions = None  # Decomposed components of the objective function
        self.num_components = None  # Number of component functions

    def initialize_agents(self):
        """ Generate initial hypotheses for agents randomly """
        self.agents = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                        (self.population_size, self.dim))
        self.activities = np.zeros(self.population_size, dtype=bool)
        # Determine number of components by evaluating objective_function on a test input
        test_input = self.agents[0]
        components = self.objective_function(test_input)
        self.num_components = len(components)  # Set based on actual number of components
        self.component_functions = lambda x, i: self.objective_function(x)[i % self.num_components]

    def evaluate_component(self, hypothesis, component_idx):
        """ Evaluate a single component function for a hypothesis """
        value = self.component_functions(hypothesis, component_idx)
        # Normalize to [0,1] range for probabilistic test
        # Assume component values are already in a suitable range or scale them
        max_value = max(1.0, abs(value))  # Avoid division by zero, adjust based on problem
        t = abs(value) / max_value
        return 0 if np.random.rand() < t else 1  # Probabilistic test outcome

    def test_phase(self):
        """ Perform the test phase: each agent evaluates a random component function """
        for i in range(self.population_size):
            component_idx = np.random.randint(0, self.num_components)
            result = self.evaluate_component(self.agents[i], component_idx)
            self.activities[i] = (result == 0)  # Active if component evaluation returns 0

    def diffusion_phase(self):
        """ Perform the diffusion phase: inactive agents recruit or pick random hypotheses """
        for i in range(self.population_size):
            if not self.activities[i]:  # Inactive agent
                agent2_idx = np.random.randint(0, self.population_size)
                if self.activities[agent2_idx]:
                    # Copy hypothesis from active agent, with possible mutation
                    new_hypothesis = self.agents[agent2_idx].copy()
                    if np.random.rand() < self.mutation_rate:
                        offset = np.random.normal(0, 1, self.dim) / self.mutation_scale
                        new_hypothesis += offset
                        new_hypothesis = np.clip(new_hypothesis, self.bounds[:, 0], self.bounds[:, 1])
                    self.agents[i] = new_hypothesis
                else:
                    # Pick a new random hypothesis
                    self.agents[i] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
            elif self.context_sensitive and self.activities[i]:
                # Context-sensitive diffusion: active agent may become inactive
                agent2_idx = np.random.randint(0, self.population_size)
                if self.activities[agent2_idx] and np.all(self.agents[agent2_idx] == self.agents[i]):
                    self.activities[i] = False
                    self.agents[i] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)

    def evaluate_full_objective(self, hypothesis):
        """ Compute the full objective function value for a hypothesis """
        return sum(self.objective_function(hypothesis))

    def check_convergence(self):
        """ Check if a cluster of sufficient size has formed at the best solution """
        if self.best_solution is None:
            return False
        distances = np.sqrt(np.sum((self.agents - self.best_solution) ** 2, axis=1))
        cluster_size = np.sum(distances < 1e-3)  # Count agents close to best solution
        return cluster_size / self.population_size >= self.cluster_threshold

    def optimize(self):
        """ Run the Stochastic Diffusion Search optimization """
        self.initialize_agents()
        for iteration in range(self.max_iter):
            # Test phase
            self.test_phase()

            # Diffusion phase
            self.diffusion_phase()

            # Evaluate best solution (full objective for tracking best)
            fitness = np.array([self.evaluate_full_objective(agent) for agent in self.agents])
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.agents[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Log progress
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

            # Check halting criterion
            if self.check_convergence():
                print(f"Converged at iteration {iteration + 1}")
                break

        return self.best_solution, self.best_value, self.history
