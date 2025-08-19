import numpy as np

class ElectromagneticFieldOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 randomization_rate=0.3, positive_selection_rate=0.2, positive_field_ratio=0.1, 
                 negative_field_ratio=0.45):
        """
        Initialize the Electromagnetic Field Optimization (EFO) algorithm.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of electromagnetic particles (solutions).
        - max_iter: Maximum number of iterations.
        - randomization_rate: Probability of randomizing an electromagnet.
        - positive_selection_rate: Probability of selecting from positive field.
        - positive_field_ratio: Proportion of particles in positive field.
        - negative_field_ratio: Proportion of particles in negative field.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.randomization_rate = randomization_rate
        self.positive_selection_rate = positive_selection_rate
        self.positive_field_ratio = positive_field_ratio
        self.negative_field_ratio = negative_field_ratio
        self.golden_ratio = (1 + np.sqrt(5)) / 2  # Golden ratio (phi)

        self.em_population = None  # Population of electromagnetic particles
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

        # Precompute random indices and values for efficiency
        self.r_index1 = np.random.randint(0, int(self.population_size * self.positive_field_ratio), 
                                         (self.dim, self.max_iter))  # Positive field indices
        self.r_index2 = np.random.randint(int(self.population_size * (1 - self.negative_field_ratio)), 
                                         self.population_size, (self.dim, self.max_iter))  # Negative field indices
        self.r_index3 = np.random.randint(int(self.population_size * self.positive_field_ratio), 
                                         int(self.population_size * (1 - self.negative_field_ratio)), 
                                         (self.dim, self.max_iter))  # Neutral field indices
        self.ps = np.random.rand(self.dim, self.max_iter)  # Selection probabilities
        self.r_force = np.random.rand(self.max_iter)  # Random force
        self.rp = np.random.rand(self.max_iter)  # Randomization probabilities
        self.randomization = np.random.rand(self.max_iter)  # Randomization coefficients

    def initialize_population(self):
        """ Generate initial electromagnetic particles randomly """
        self.em_population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                              (self.population_size, self.dim))
        self.evaluate_and_sort_population()

    def evaluate_and_sort_population(self):
        """ Compute fitness values and sort population by fitness """
        fitness = np.array([self.objective_function(particle) for particle in self.em_population])
        # Combine particles with their fitness values
        population_with_fitness = np.column_stack((self.em_population, fitness))
        # Sort by fitness (ascending)
        sorted_indices = np.argsort(fitness)
        self.em_population = population_with_fitness[sorted_indices, :-1]
        fitness = fitness[sorted_indices]
        # Update best solution
        if fitness[0] < self.best_value:
            self.best_solution = self.em_population[0].copy()
            self.best_value = fitness[0]

    def generate_new_particle(self, generation):
        """ Generate a new particle based on EFO rules """
        new_particle = np.zeros(self.dim)
        r = self.r_force[generation]

        for i in range(self.dim):
            if self.ps[i, generation] > self.positive_selection_rate:
                # Use particles from positive, neutral, and negative fields
                new_particle[i] = (self.em_population[self.r_index3[i, generation], i] +
                                  self.golden_ratio * r * (self.em_population[self.r_index1[i, generation], i] -
                                                          self.em_population[self.r_index3[i, generation], i]) +
                                  r * (self.em_population[self.r_index3[i, generation], i] -
                                       self.em_population[self.r_index2[i, generation], i]))
            else:
                # Copy from positive field
                new_particle[i] = self.em_population[self.r_index1[i, generation], i]

            # Check boundaries
            if new_particle[i] < self.bounds[i, 0] or new_particle[i] > self.bounds[i, 1]:
                new_particle[i] = self.bounds[i, 0] + (self.bounds[i, 1] - self.bounds[i, 0]) * \
                                  self.randomization[generation]

        # Randomize one dimension with probability randomization_rate
        if self.rp[generation] < self.randomization_rate:
            ri = np.random.randint(0, self.dim)
            new_particle[ri] = self.bounds[ri, 0] + (self.bounds[ri, 1] - self.bounds[ri, 0]) * \
                               self.randomization[generation]

        return new_particle

    def insert_particle(self, new_particle):
        """ Insert new particle into population if it improves fitness """
        new_fitness = self.objective_function(new_particle)
        if new_fitness < self.objective_function(self.em_population[-1]):
            # Find insertion position
            fitness = np.array([self.objective_function(p) for p in self.em_population])
            insert_pos = np.searchsorted(fitness, new_fitness)
            # Shift population and insert new particle
            self.em_population = np.vstack((self.em_population[:insert_pos], 
                                            new_particle, 
                                            self.em_population[insert_pos:-1]))

    def optimize(self):
        """ Run the Electromagnetic Field Optimization algorithm """
        self.initialize_population()
        generation = 0

        while generation < self.max_iter:
            # Generate and evaluate new particle
            new_particle = self.generate_new_particle(generation)
            self.insert_particle(new_particle)
            self.evaluate_and_sort_population()

            self.history.append((generation, self.best_solution.copy(), self.best_value))
            if generation % 1000 == 0:  # Print progress periodically
                print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

            generation += 1

        return self.best_solution, self.best_value, self.history
