import numpy as np

class GizaPyramidsConstruction:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, G=9.8, theta=np.radians(14), mu_min=0.3, mu_max=0.5, V_min=0.1, V_max=2.0):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.max_iter = max_iter
        self.G = G  # Gravity constant
        self.theta = theta  # Ramp angle (converted to radians)
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.V_min = V_min
        self.V_max = V_max
        self.pyramids = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []
    
    def initialize_pyramids(self):
        """ Generate an initial random population """
        self.pyramids = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim))
    
    def evaluate_pyramids(self):
        """ Compute fitness values """
        return np.array([self.objective_function(p) for p in self.pyramids])
    
    def compute_movement(self, velocity, friction):
        """ Compute the stone and worker movement distances """
        d = (velocity ** 2) / (2 * self.G * (np.sin(self.theta) + friction * np.cos(self.theta)))  # Stone movement
        x = (velocity ** 2) / (2 * self.G * np.sin(self.theta))  # Worker movement
        return d, x
    
    def update_pyramids(self):
        """ Perform physics-based movement updates """
        fitness = self.evaluate_pyramids()
        
        print(f"Before Update: {fitness[:5]}")  # Print first 5 fitness values for debugging
        
        best_index = np.argmin(fitness)
        best_pyramid = self.pyramids[best_index]
        
        for i in range(self.population_size):
            if i != best_index:
                velocity = np.random.uniform(self.V_min, self.V_max, self.dim)
                friction = np.random.uniform(self.mu_min, self.mu_max, self.dim)
                d, x = self.compute_movement(velocity, friction)

                epsilon = np.random.uniform(-0.5 * (self.V_max - self.V_min), 0.5 * (self.V_max - self.V_min), self.dim)
                self.pyramids[i] = (self.pyramids[i] + d) * (x * epsilon)
                self.pyramids[i] = np.clip(self.pyramids[i], self.bounds[:, 0], self.bounds[:, 1])

        fitness_after = self.evaluate_pyramids()
        print(f"After Update: {fitness_after[:5]}")  # Print first 5 fitness values after update
    
    def optimize(self):
        """ Run the GPC algorithm """
        self.initialize_pyramids()
        for generation in range(self.max_iter):
            self.update_pyramids()
            fitness = self.evaluate_pyramids()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.pyramids[min_idx].copy()
                self.best_value = fitness[min_idx]
            
            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")
        
        return self.best_solution, self.best_value, self.history
