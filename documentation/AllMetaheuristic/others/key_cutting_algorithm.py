import numpy as np

class KeyCuttingAlgorithm:
    def __init__(self, objective_function, num_buses, num_branches, population_size=20, max_iter=100, 
                 probability_threshold=0.5):
        """
        Initialize the Key Cutting Algorithm (KCA) for measurement placement.

        Parameters:
        - objective_function: Function to evaluate fitness (based on observability and critical measurements).
        - num_buses: Number of buses in the power system.
        - num_branches: Number of branches in the power system.
        - population_size: Number of keys (solutions) in the population.
        - max_iter: Maximum number of iterations.
        - probability_threshold: Threshold for key tooth modification (KCA1: > threshold, KCA2: < threshold).
        """
        self.objective_function = objective_function
        self.num_buses = num_buses
        self.num_branches = num_branches
        self.key_length = num_buses + num_branches  # Total key teeth (buses + branches)
        self.population_size = population_size
        self.max_iter = max_iter
        self.probability_threshold = probability_threshold

        self.keys = None  # Population of keys (binary solutions)
        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    def initialize_keys(self):
        """ Generate initial key set randomly with binary values (0 or 1) """
        self.keys = np.random.randint(0, 2, (self.population_size, self.key_length))

    def evaluate_keys(self):
        """ Compute fitness values for all keys """
        return np.array([self.objective_function(key, self.num_buses) for key in self.keys])

    def calculate_probability_factor(self):
        """ Calculate probability factor for each key tooth """
        probability_factors = np.zeros((self.population_size // 2, self.key_length))
        selected_keys = self.keys[:self.population_size // 2]  # First half with higher fitness
        for j in range(self.key_length):
            tooth_column = selected_keys[:, j]
            probability_factors[:, j] = 1 - (np.sum(tooth_column) / (self.population_size // 2))
        return probability_factors

    def generate_new_keys(self, probability_factors, use_kca1=True):
        """ Generate second half of keys based on probability factors """
        new_keys = self.keys[:self.population_size // 2].copy()  # Copy first half
        for i in range(self.population_size // 2):
            for j in range(self.key_length):
                random_num = np.random.rand()
                condition = random_num > probability_factors[i, j] if use_kca1 else random_num < probability_factors[i, j]
                if condition:
                    new_keys[i, j] = 1 - new_keys[i, j]  # Flip the key tooth (0 to 1 or 1 to 0)
        self.keys[self.population_size // 2:] = new_keys

    def optimize(self, use_kca1=True):
        """
        Run the Key Cutting Algorithm (KCA) optimization.

        Parameters:
        - use_kca1: If True, use KCA1 (random_num > probability_factor); if False, use KCA2.
        """
        self.initialize_keys()
        for generation in range(self.max_iter):
            # Step 1: Evaluate fitness
            fitness = self.evaluate_keys()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_fitness:
                self.best_solution = self.keys[min_idx].copy()
                self.best_fitness = fitness[min_idx]

            # Step 2: Select top 50% keys
            sorted_indices = np.argsort(fitness)
            self.keys = self.keys[sorted_indices]  # Sort keys by fitness

            # Step 3: Calculate probability factors
            probability_factors = self.calculate_probability_factor()

            # Step 4: Generate new keys for second half
            self.generate_new_keys(probability_factors, use_kca1)

            # Step 5: Log history
            self.history.append((generation, self.best_solution.copy(), self.best_fitness))
            print(f"Iteration {generation + 1}: Best Fitness = {self.best_fitness}")

        return self.best_solution, self.best_fitness, self.history

def measurement_placement_fitness(key, num_buses):
    """
    Fitness function for measurement placement, incorporating observability and critical measurement constraints.

    Parameters:
    - key: Binary array representing measurement placement (1 for installed, 0 for not installed).
    - num_buses: Number of buses in the power system.

    Returns:
    - Fitness value based on observability, number of measurements, and critical measurements.
    """
    # Simulate measurement matrix H and observability check
    # For simplicity, assume rank(H) is computed externally or approximated
    rank_H = min(np.sum(key), num_buses - 1)  # Simplified rank approximation
    M = np.sum(key)  # Number of measurements
    CM = 0  # Assume critical measurement check is external; set to 0 for simplicity

    # Fitness function: (N - rank(H)) + (M - N) + CM * N
    fitness = (num_buses - rank_H) + (M - num_buses) + CM * num_buses
    return fitness

# Example usage
if __name__ == "__main__":
    # Example for a 10-bus system with 13 branches (as per paper's 10-bus system)
    num_buses = 10
    num_branches = 13
    kca = KeyCuttingAlgorithm(
        objective_function=measurement_placement_fitness,
        num_buses=num_buses,
        num_branches=num_branches,
        population_size=20,
        max_iter=50,
        probability_threshold=0.5
    )

    # Run KCA1
    best_solution, best_fitness, history = kca.optimize(use_kca1=True)
    print(f"KCA1 - Best Solution: {best_solution}, Best Fitness: {best_fitness}")

    # Reset and run KCA2
    kca = KeyCuttingAlgorithm(
        objective_function=measurement_placement_fitness,
        num_buses=num_buses,
        num_branches=num_branches,
        population_size=20,
        max_iter=50,
        probability_threshold=0.5
    )
    best_solution, best_fitness, history = kca.optimize(use_kca1=False)
    print(f"KCA2 - Best Solution: {best_solution}, Best Fitness: {best_fitness}")
