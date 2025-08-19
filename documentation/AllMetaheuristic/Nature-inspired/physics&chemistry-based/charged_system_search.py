import numpy as np

class ChargedSystemSearch:
    def __init__(self, objective_function, dim, bounds, population_size=16, max_iter=100, 
                 ka=1.0, kv=1.0, a=1.0, epsilon=1e-10):
        """
        Initialize the Charged System Search (CSS) optimizer.

        Parameters:
        - objective_function: Function to optimize (e.g., ECBI function for damage detection).
        - dim: Number of dimensions (variables, e.g., number of structural elements).
        - bounds: Tuple of (lower, high) bounds for each dimension.
        - population_size: Number of charged particles (CPs).
        - max_iter: Maximum number of iterations.
        - ka: Acceleration coefficient.
        - kv: Velocity coefficient.
        - a: Distance threshold for force calculation.
        - epsilon: Small positive number to avoid division by zero.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.ka = ka
        self.kv = kv
        self.a = a
        self.epsilon = epsilon

        self.charged_particles = None  # Population of charged particles (solutions)
        self.velocities = None  # Velocities of charged particles
        self.best_solution = None
        self.best_value = float("inf")
        self.charged_memory = []  # Stores best solutions
        self.cm_size = population_size // 4  # Size of charged memory
        self.history = []

    def initialize_charged_particles(self):
        """Generate initial charged particles and velocities randomly."""
        self.charged_particles = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                                  (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))

    def evaluate_charged_particles(self):
        """Compute fitness values for the charged particles."""
        return np.array([self.objective_function(cp) for cp in self.charged_particles])

    def calculate_charge(self, fitness):
        """Calculate charge magnitude for each charged particle."""
        fitworst = np.max(fitness)
        fitbest = np.min(fitness)
        if fitbest == fitworst:
            return np.ones(self.population_size)
        return (fitness - fitworst) / (fitbest - fitworst)

    def calculate_forces(self, fitness):
        """Calculate resultant forces on each charged particle."""
        charges = self.calculate_charge(fitness)
        forces = np.zeros((self.population_size, self.dim))
        
        for j in range(self.population_size):
            force = np.zeros(self.dim)
            for i in range(self.population_size):
                if i != j:
                    # Calculate separation distance
                    r_ij = np.linalg.norm(self.charged_particles[i] - self.charged_particles[j])
                    r_ij_norm = r_ij / (np.linalg.norm((self.charged_particles[i] + self.charged_particles[j]) / 2 - 
                                                       (self.best_solution if self.best_solution is not None else self.charged_particles[0])) + self.epsilon)
                    
                    # Probability of attraction
                    p_ij = 1 if (fitness[i] < fitness[j] or 
                                 (fitness[i] - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + self.epsilon) > np.random.rand()) else 0
                    
                    # Force calculation based on distance
                    if r_ij < self.a:
                        force_term = (charges[i] / self.a**3) * r_ij
                    else:
                        force_term = (charges[i] / r_ij**2)
                    
                    force += p_ij * force_term * (self.charged_particles[i] - self.charged_particles[j])
            
            forces[j] = charges[j] * force
        
        return forces

    def update_positions(self, forces):
        """Update positions and velocities of charged particles."""
        dt = 1.0  # Time step
        for j in range(self.population_size):
            # Update velocity
            self.velocities[j] = np.random.rand() * self.kv * self.velocities[j] + \
                                np.random.rand() * self.ka * forces[j] / 1.0  # Mass assumed as 1
            
            # Update position
            self.charged_particles[j] += self.velocities[j] * dt
            
            # Apply harmony search-based boundary handling
            self.charged_particles[j] = np.clip(self.charged_particles[j], 
                                               self.bounds[:, 0], self.bounds[:, 1])

    def update_charged_memory(self, fitness):
        """Update charged memory with best solutions."""
        sorted_indices = np.argsort(fitness)
        for i in sorted_indices[:self.cm_size]:
            if len(self.charged_memory) < self.cm_size:
                self.charged_memory.append((self.charged_particles[i].copy(), fitness[i]))
            else:
                worst_cm_idx = np.argmax([cm[1] for cm in self.charged_memory])
                if fitness[i] < self.charged_memory[worst_cm_idx][1]:
                    self.charged_memory[worst_cm_idx] = (self.charged_particles[i].copy(), fitness[i])

    def optimize(self):
        """Run the Charged System Search optimization."""
        self.initialize_charged_particles()
        
        for iteration in range(self.max_iter):
            fitness = self.evaluate_charged_particles()
            min_idx = np.argmin(fitness)
            
            # Update best solution
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.charged_particles[min_idx].copy()
                self.best_value = fitness[min_idx]
            
            # Calculate forces and update positions
            forces = self.calculate_forces(fitness)
            self.update_positions(forces)
            
            # Update charged memory
            self.update_charged_memory(fitness)
            
            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")
        
        return self.best_solution, self.best_value, self.history

# Example usage for structural damage detection
if __name__ == "__main__":
    def ecbi_objective_function(X):
        """
        Example ECBI objective function for structural damage detection.
        X: Damage variables (reduction in stiffness for each element).
        Returns: ECBI value to be minimized.
        """
        # Placeholder: Replace with actual modal analysis and ECBI calculation
        # For demonstration, assume a simple quadratic function
        return np.sum(X**2)  # Dummy objective function

    # Define problem parameters
    dim = 10  # Number of elements (e.g., for a 10-element cantilever beam)
    bounds = [(0, 1)] * dim  # Damage variables between 0 and 1
    css = ChargedSystemSearch(ecbi_objective_function, dim, bounds, 
                              population_size=16, max_iter=3000)
    
    # Run optimization
    best_solution, best_value, history = css.optimize()
    print(f"Best Solution: {best_solution}")
    print(f"Best Value: {best_value}")
