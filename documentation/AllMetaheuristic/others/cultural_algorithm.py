import numpy as np

class CulturalAlgorithmOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=1000, 
                 p_accept=0.35, alpha=0.3, beta=0.5):
        """
        Initialize the Cultural Algorithm optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of solutions in the population.
        - max_iter: Maximum number of iterations.
        - p_accept: Acceptance ratio for selecting individuals to update culture.
        - alpha: Scaling factor for normative influence.
        - beta: Scaling factor for normative range adjustment (unused in method 3).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.p_accept = p_accept
        self.n_accept = round(p_accept * population_size)  # Number of accepted individuals
        self.alpha = alpha
        self.beta = beta

        self.population = None  # Population of solutions
        self.costs = None  # Costs (fitness values) of solutions
        self.best_solution = None
        self.best_cost = float("inf")
        self.history = []

        # Initialize culture
        self.culture = {
            "Situational": {"Position": None, "Cost": float("inf")},
            "Normative": {
                "Min": np.full(dim, float("inf")),
                "Max": np.full(dim, -float("inf")),
                "L": np.full(dim, float("inf")),
                "U": np.full(dim, float("inf")),
                "Size": np.zeros(dim)
            }
        }

    def initialize_population(self):
        """Generate initial population randomly within bounds."""
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                            (self.population_size, self.dim))
        self.costs = self.evaluate_population()
        
        # Sort population by cost
        sort_indices = np.argsort(self.costs)
        self.population = self.population[sort_indices]
        self.costs = self.costs[sort_indices]
        
        # Initialize culture with top n_accept individuals
        selected_pop = self.population[:self.n_accept]
        selected_costs = self.costs[:self.n_accept]
        self.adjust_culture(selected_pop, selected_costs)
        
        # Set initial best solution
        self.best_solution = self.culture["Situational"]["Position"].copy()
        self.best_cost = self.culture["Situational"]["Cost"]

    def evaluate_population(self):
        """Compute fitness values for the population."""
        return np.array([self.objective_function(ind) for ind in self.population])

    def adjust_culture(self, selected_pop, selected_costs):
        """Update situational and normative components of the culture."""
        for i in range(len(selected_pop)):
            # Update Situational component
            if selected_costs[i] < self.culture["Situational"]["Cost"]:
                self.culture["Situational"]["Position"] = selected_pop[i].copy()
                self.culture["Situational"]["Cost"] = selected_costs[i]
            
            # Update Normative component
            for j in range(self.dim):
                if (selected_pop[i][j] < self.culture["Normative"]["Min"][j] or
                    selected_costs[i] < self.culture["Normative"]["L"][j]):
                    self.culture["Normative"]["Min"][j] = selected_pop[i][j]
                    self.culture["Normative"]["L"][j] = selected_costs[i]
                if (selected_pop[i][j] > self.culture["Normative"]["Max"][j] or
                    selected_costs[i] < self.culture["Normative"]["U"][j]):
                    self.culture["Normative"]["Max"][j] = selected_pop[i][j]
                    self.culture["Normative"]["U"][j] = selected_costs[i]
        
        # Update Normative Size
        self.culture["Normative"]["Size"] = (self.culture["Normative"]["Max"] - 
                                            self.culture["Normative"]["Min"])

    def influence_culture(self):
        """Apply cultural influence to update population positions (Method 3)."""
        new_population = self.population.copy()
        for i in range(self.population_size):
            for j in range(self.dim):
                # Compute standard deviation
                sigma = self.alpha * self.culture["Normative"]["Size"][j]
                dx = sigma * np.random.randn()
                
                # Adjust direction based on Situational component
                if new_population[i][j] < self.culture["Situational"]["Position"][j]:
                    dx = abs(dx)
                elif new_population[i][j] > self.culture["Situational"]["Position"][j]:
                    dx = -abs(dx)
                
                # Update position
                new_population[i][j] += dx
        
        # Clip to bounds
        new_population = np.clip(new_population, self.bounds[:, 0], self.bounds[:, 1])
        self.population = new_population

    def optimize(self):
        """Run the Cultural Algorithm optimization."""
        self.initialize_population()
        
        for iteration in range(self.max_iter):
            # Apply cultural influence
            self.influence_culture()
            
            # Evaluate new costs
            self.costs = self.evaluate_population()
            
            # Sort population by cost
            sort_indices = np.argsort(self.costs)
            self.population = self.population[sort_indices]
            self.costs = self.costs[sort_indices]
            
            # Adjust culture with top n_accept individuals
            selected_pop = self.population[:self.n_accept]
            selected_costs = self.costs[:self.n_accept]
            self.adjust_culture(selected_pop, selected_costs)
            
            # Update best solution
            if self.culture["Situational"]["Cost"] < self.best_cost:
                self.best_solution = self.culture["Situational"]["Position"].copy()
                self.best_cost = self.culture["Situational"]["Cost"]
            
            # Store history
            self.history.append((iteration, self.best_solution.copy(), self.best_cost))
            print(f"Iteration {iteration + 1}: Best Cost = {self.best_cost}")
        
        return self.best_solution, self.best_cost, self.history
