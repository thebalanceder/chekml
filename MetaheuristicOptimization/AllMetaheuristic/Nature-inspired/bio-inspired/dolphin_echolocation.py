import numpy as np
import uuid

class DolphinEcholocationOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_loops=100,
                 convergence_power=1.0, effective_radius_factor=0.25, probability_threshold=0.1):
        """
        Initialize the Dolphin Echolocation Algorithm (DEA) optimizer.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of dolphin locations (solutions).
        - max_loops: Maximum number of iterations (loops).
        - convergence_power: Power parameter for the convergence curve (Power in PDF).
        - effective_radius_factor: Factor to determine effective radius (< 1/4 search space).
        - probability_threshold: Minimum probability threshold for location selection.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_loops = max_loops
        self.convergence_power = convergence_power
        self.effective_radius_factor = effective_radius_factor
        self.probability_threshold = probability_threshold

        self.locations = None  # Population of dolphin locations (solutions)
        self.best_location = None
        self.best_value = float("inf")
        self.history = []
        self.effective_radius = None
        self.accumulative_fitness = None
        self.alternatives = None

    def initialize_locations(self):
        """ Generate initial dolphin locations randomly """
        self.locations = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                          (self.population_size, self.dim))
        # Calculate effective radius based on search space size
        search_space_size = np.abs(self.bounds[:, 1] - self.bounds[:, 0])
        self.effective_radius = self.effective_radius_factor * np.min(search_space_size)
        # Initialize alternatives matrix (discretized search space per dimension)
        self.alternatives = [np.linspace(self.bounds[i, 0], self.bounds[i, 1], 100) for i in range(self.dim)]

    def evaluate_locations(self):
        """ Compute fitness values for the dolphin locations """
        return np.array([self.objective_function(loc) for loc in self.locations])

    def calculate_accumulative_fitness(self, fitness):
        """
        Calculate accumulative fitness for alternatives based on dolphin rules.

        Distributes fitness to alternatives and their neighbors within effective radius.
        """
        self.accumulative_fitness = [np.zeros(len(alt)) for alt in self.alternatives]
        for i in range(self.population_size):
            for j in range(self.dim):
                # Find the closest alternative index for the current location in dimension j
                loc_value = self.locations[i, j]
                alt_idx = np.argmin(np.abs(self.alternatives[j] - loc_value))
                # Distribute fitness to the alternative and its neighbors
                fit = fitness[i]
                for k in range(-10, 11):  # As per PDF pseudo-code
                    neighbor_idx = alt_idx + k
                    if 0 <= neighbor_idx < len(self.alternatives[j]):
                        weight = (10 - abs(k)) / 10.0  # Linear decay
                        self.accumulative_fitness[j][neighbor_idx] += weight * fit

    def get_convergence_probability(self, loop):
        """
        Compute the predefined probability curve value for the current loop.

        Uses a power-based convergence curve as recommended in the PDF.
        """
        t = loop / self.max_loops
        return t ** self.convergence_power

    def update_probabilities(self, loop):
        """
        Allocate probabilities to alternatives based on accumulative fitness and convergence curve.

        The best location gets the convergence curve probability, others are distributed proportionally.
        """
        fitness = self.evaluate_locations()
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_value:
            self.best_location = self.locations[min_idx].copy()
            self.best_value = fitness[min_idx]

        self.calculate_accumulative_fitness(fitness)
        probabilities = [np.zeros(len(alt)) for alt in self.alternatives]
        convergence_prob = self.get_convergence_probability(loop)

        # Find the best alternative indices for the best location
        best_alt_indices = []
        for j in range(self.dim):
            loc_value = self.best_location[j]
            alt_idx = np.argmin(np.abs(self.alternatives[j] - loc_value))
            best_alt_indices.append(alt_idx)

        # Assign probabilities
        for j in range(self.dim):
            total_af = np.sum(self.accumulative_fitness[j])
            if total_af == 0:
                probabilities[j] = np.ones(len(self.alternatives[j])) / len(self.alternatives[j])
            else:
                # Best alternative gets convergence probability
                probabilities[j][best_alt_indices[j]] = convergence_prob
                remaining_prob = 1.0 - convergence_prob
                # Distribute remaining probability based on accumulative fitness
                for k in range(len(self.alternatives[j])):
                    if k != best_alt_indices[j]:
                        prob = (self.accumulative_fitness[j][k] / total_af) * remaining_prob if total_af > 0 else 0
                        probabilities[j][k] = max(prob, self.probability_threshold)

                # Normalize probabilities to sum to 1
                prob_sum = np.sum(probabilities[j])
                if prob_sum > 0:
                    probabilities[j] /= prob_sum

        return probabilities

    def select_new_locations(self, probabilities):
        """ Select new locations for the next loop based on calculated probabilities """
        new_locations = np.zeros_like(self.locations)
        for i in range(self.population_size):
            for j in range(self.dim):
                # Select an alternative based on probabilities
                try:
                    selected_idx = np.random.choice(len(self.alternatives[j]), p=probabilities[j])
                    new_locations[i, j] = self.alternatives[j][selected_idx]
                except ValueError:
                    # In case of invalid probabilities, select randomly
                    new_locations[i, j] = np.random.choice(self.alternatives[j])
        self.locations = np.clip(new_locations, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """ Run the Dolphin Echolocation Algorithm """
        self.initialize_locations()
        for loop in range(self.max_loops):
            # Update probabilities and select new locations
            probabilities = self.update_probabilities(loop)
            self.select_new_locations(probabilities)

            # Log history
            self.history.append((loop, self.best_location.copy(), self.best_value))
            print(f"Loop {loop + 1}: Best Value = {self.best_value}")

            # Check termination criteria (e.g., convergence or max loops)
            if loop > 0 and abs(self.history[loop][2] - self.history[loop-1][2]) < 1e-6:
                print("Convergence reached.")
                break

        return self.best_location, self.best_value, self.history

