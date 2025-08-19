import numpy as np

class EmperorPenguinOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 f=2.0, l=1.5, M=0.5, adaptation_interval=10):
        """
        Initialize the Self-adaptive Emperor Penguin Optimizer (SA-EPO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of penguins (solutions).
        - max_iter: Maximum number of iterations.
        - f: Temperature profile parameter (initial value).
        - l: Distance profile parameter (initial value).
        - M: Social forces parameter (initial value).
        - adaptation_interval: Interval for updating strategy probabilities.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.f = f
        self.l = l
        self.M = M
        self.adaptation_interval = adaptation_interval

        self.penguins = None  # Population of penguins (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []
        
        # Parameter adaptation strategies and their probabilities
        self.strategies = ['linear', 'exponential', 'chaotic']
        self.strategy_probs = np.array([1.0/3, 1.0/3, 1.0/3])
        self.strategy_success = np.zeros(3)  # Track success of each strategy
        
    def initialize_penguins(self):
        """ Generate initial penguin positions randomly """
        self.penguins = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                        (self.population_size, self.dim))

    def evaluate_penguins(self):
        """ Compute fitness values for the penguin positions """
        return np.array([self.objective_function(penguin) for penguin in self.penguins])

    def compute_huddle_boundary(self, t):
        """ Compute huddle boundary and temperature profile """
        T = self.max_iter
        T_prime = 2 - (t / T)  # Temperature profile
        R = np.random.rand()  # Random factor for boundary
        return T_prime, R

    def adapt_parameters(self, strategy, t):
        """ Adapt control parameters (f, l, M) based on selected strategy """
        T = self.max_iter
        if strategy == 'linear':
            self.f = 2.0 - (t / T) * 1.5
            self.l = 1.5 - (t / T) * 1.0
            self.M = 0.5 + (t / T) * 0.3
        elif strategy == 'exponential':
            self.f = 2.0 * np.exp(-t / (T / 2))
            self.l = 1.5 * np.exp(-t / (T / 3))
            self.M = 0.5 * (1 + np.tanh(t / (T / 4)))
        elif strategy == 'chaotic':
            # Logistic map for chaotic adaptation
            x = 0.7  # Initial value
            for _ in range(t % 10):  # Simple chaotic iteration
                x = 4 * x * (1 - x)
            self.f = 1.5 + x * 0.5
            self.l = 1.0 + x * 0.5
            self.M = 0.3 + x * 0.4
        return self.f, self.l, self.M

    def update_strategy_probabilities(self):
        """ Update strategy selection probabilities based on historical success """
        total_success = np.sum(self.strategy_success) + 1e-10  # Avoid division by zero
        self.strategy_probs = self.strategy_success / total_success
        self.strategy_probs = np.clip(self.strategy_probs, 0.1, 0.9)  # Ensure diversity
        self.strategy_probs /= np.sum(self.strategy_probs)  # Normalize
        self.strategy_success *= 0.9  # Decay success to favor recent performance

    def huddle_movement(self, index, t):
        """ Simulate penguin movement in huddle """
        T_prime, R = self.compute_huddle_boundary(t)
        
        # Select adaptation strategy
        strategy = np.random.choice(self.strategies, p=self.strategy_probs)
        f, l, M = self.adapt_parameters(strategy, t)
        
        # Compute distance to best solution
        if self.best_solution is not None:
            D = np.abs(f * np.random.rand() * self.best_solution - self.penguins[index])
        else:
            D = np.abs(f * np.random.rand() * self.penguins[index])
        
        # Compute social forces
        S = M * np.exp(-t / l) - np.exp(-t)
        
        # Update position
        new_solution = self.penguins[index] + S * D * np.random.rand(self.dim)
        
        # Clip to bounds
        new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
        
        # Evaluate new solution
        new_value = self.objective_function(new_solution)
        old_value = self.objective_function(self.penguins[index])
        
        # Track strategy success
        if new_value < old_value:
            strategy_idx = self.strategies.index(strategy)
            self.strategy_success[strategy_idx] += 1
        
        return new_solution

    def optimize(self):
        """ Run the Self-adaptive Emperor Penguin Optimization """
        self.initialize_penguins()
        for generation in range(self.max_iter):
            fitness = self.evaluate_penguins()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.penguins[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Update penguin positions
            for i in range(self.population_size):
                self.penguins[i] = self.huddle_movement(i, generation)

            # Update strategy probabilities periodically
            if generation % self.adaptation_interval == 0 and generation > 0:
                self.update_strategy_probabilities()

            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
