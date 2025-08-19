import numpy as np

class TeachingLearningBasedOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100):
        """
        Initialize the TLBO optimizer.

        Parameters:
        - objective_function: Function to optimize (minimization problem).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of students (solutions) in the population.
        - max_iter: Maximum number of iterations (generations).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.population = None  # Population of students (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []  # Tracks best fitness value per iteration

    def initialize_population(self):
        """ Generate initial population randomly within bounds """
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def evaluate_population(self):
        """ Compute fitness values for the population """
        return np.array([self.objective_function(individual) for individual in self.population])

    def teacher_phase(self, index, fitness):
        """ Simulate the teacher phase for the given student index """
        # Compute mean of the population
        mean_student = np.mean(self.population, axis=0)
        
        # Identify the teacher (best solution)
        best_idx = np.argmin(fitness)
        teacher = self.population[best_idx]
        
        # Random teaching factor (1 or 2)
        tf = np.random.randint(1, 3)
        
        # Generate new solution
        new_solution = self.population[index] + np.random.rand(self.dim) * (teacher - tf * mean_student)
        
        # Bound the solution
        new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
        
        # Evaluate new solution
        new_fitness = self.objective_function(new_solution)
        
        # Greedy selection
        if new_fitness < fitness[index]:
            self.population[index] = new_solution
            fitness[index] = new_fitness
        
        return fitness

    def learner_phase(self, index, fitness, partner_idx):
        """ Simulate the learner phase for the given student index """
        # Generate new solution based on comparison with partner
        if fitness[index] < fitness[partner_idx]:
            new_solution = self.population[index] + np.random.rand(self.dim) * (self.population[index] - self.population[partner_idx])
        else:
            new_solution = self.population[index] + np.random.rand(self.dim) * (self.population[partner_idx] - self.population[index])
        
        # Bound the solution
        new_solution = np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])
        
        # Evaluate new solution
        new_fitness = self.objective_function(new_solution)
        
        # Greedy selection
        if new_fitness < fitness[index]:
            self.population[index] = new_solution
            fitness[index] = new_fitness
        
        return fitness

    def optimize(self):
        """ Run the Teaching-Learning-Based Optimization """
        # Initialize population
        self.initialize_population()
        
        # Evaluate initial population
        fitness = self.evaluate_population()
        
        for generation in range(self.max_iter):
            # Partner selection using random permutation
            partners = np.random.permutation(self.population_size)
            
            # Process each student
            for i in range(self.population_size):
                # Teacher phase
                fitness = self.teacher_phase(i, fitness)
                
                # Learner phase
                fitness = self.learner_phase(i, fitness, partners[i])
            
            # Track best solution and fitness
            min_idx = np.argmin(fitness)
            current_best_value = fitness[min_idx]
            if current_best_value < self.best_value:
                self.best_solution = self.population[min_idx].copy()
                self.best_value = current_best_value
            
            # Store history
            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")
        
        return self.best_solution, self.best_value, self.history
