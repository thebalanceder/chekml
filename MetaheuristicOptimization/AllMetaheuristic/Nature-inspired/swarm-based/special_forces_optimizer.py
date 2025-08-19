import numpy as np

class SpecialForcesOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=30, max_iter=200, 
                 tv1=0.5, tv2=0.3, p0=0.25, k=0.4):
        """
        Initialize the Special Forces Algorithm (SFA) optimizer.

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of special force members (solutions).
        - max_iter: Maximum number of iterations.
        - tv1, tv2: Thresholds for phase transitions (exploration, transition, exploitation).
        - p0: Initial probability of losing contact.
        - k: Constant for unmanned search range (0 < k < 0.5).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.tv1 = tv1
        self.tv2 = tv2
        self.p0 = p0
        self.k = k

        self.members = None  # Population of special force members (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_members(self):
        """ Generate initial positions for special force members randomly """
        self.members = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                        (self.population_size, self.dim))

    def evaluate_members(self):
        """ Compute fitness values for the special force members """
        return np.array([self.objective_function(member) for member in self.members])

    def calculate_instruction(self, t):
        """ Calculate Instruction(t) as per Eq. (1) """
        return (1 - 0.15 * np.random.rand()) * (1 - t / self.max_iter)

    def calculate_loss_probability(self, t):
        """ Calculate probability of losing contact p(t) as per Eq. (3) """
        return self.p0 * np.cos(np.pi * t / (2 * self.max_iter))

    def calculate_raid_coefficient(self, t):
        """ Calculate raid coefficient w(t) as per Eq. (7) """
        return 0.75 - 0.55 * np.arctan((t / self.max_iter) ** (2 * np.pi))

    def large_scale_search(self, index, r1):
        """ Implement large-scale search strategy as per Eq. (4) """
        if r1 >= 0.5:
            lb, ub = self.bounds[:, 0], self.bounds[:, 1]
            term1 = r1 * (self.best_solution - self.members[index])
            term2 = (1 - r1) * (ub - lb) * np.random.choice([-1, 1], size=self.dim)
            new_position = self.members[index] + term1 + term2
            return np.clip(new_position, lb, ub)
        return None

    def raid(self, index, r1, w):
        """ Implement raid strategy as per Eq. (5, 6) """
        if r1 < 0.5:
            # Select a random member as the 'aim' (best known position for this member)
            available_indices = [i for i in range(self.population_size) if i != index]
            if not available_indices:
                return None
            aim_idx = np.random.choice(available_indices)
            X_i = self.members[index]
            X_aim = self.members[aim_idx]
            f_i = self.objective_function(X_i)
            f_aim = self.objective_function(X_aim)
            
            # Calculate search vector A_i(t) as per Eq. (6)
            A_i = (f_i / (f_i + f_aim)) * (X_aim - X_i) if f_i + f_aim != 0 else np.zeros(self.dim)
            
            # Update position as per Eq. (5)
            new_position = X_i + w * A_i
            return np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
        return None

    def transition_phase(self, index, r2, w, instruction):
        """ Implement transition phase strategy as per Eq. (8) """
        if r2 >= 0.5:
            # Reuse raid strategy
            A_i = self.raid(index, r1=0.4, w=w)  # Force raid by setting r1 < 0.5
            if A_i is not None:
                return A_i
            return self.members[index]
        else:
            new_position = instruction * (self.best_solution - self.members[index]) + 0.1 * self.members[index]
            return np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

    def arrest_rescue(self, t):
        """ Implement arrest-rescue strategy as per Eq. (9, 10) """
        X_ave = np.mean(self.members, axis=0)  # Eq. (10)
        r = np.random.uniform(-1, 1, self.dim)
        new_positions = self.best_solution + r * np.abs(self.best_solution - X_ave)
        return np.clip(new_positions, self.bounds[:, 0], self.bounds[:, 1])

    def unmanned_search(self, t):
        """ Implement unmanned search as per Eq. (11, 12) """
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        c = self.k * (lb + (1 - t / self.max_iter) * (ub - lb))  # Eq. (12)
        
        # Generate random vector v such that sum of squares equals c^2
        v = np.random.randn(self.dim)
        v = v / np.sqrt(np.sum(v**2)) * c  # Normalize to magnitude c
        
        # Select a random member's position as base
        base_idx = np.random.randint(self.population_size)
        X_u = self.members[base_idx] + v
        return np.clip(X_u, lb, ub)

    def optimize(self):
        """ Run the Special Forces Algorithm optimization """
        self.initialize_members()
        for t in range(self.max_iter):
            # Evaluate fitness and update best solution
            fitness = self.evaluate_members()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.members[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Unmanned search
            X_u = self.unmanned_search(t)
            fitness_u = self.objective_function(X_u)
            if fitness_u < self.best_value:
                self.best_solution = X_u.copy()
                self.best_value = fitness_u

            # Calculate iteration parameters
            instruction = self.calculate_instruction(t)
            p = self.calculate_loss_probability(t)
            w = self.calculate_raid_coefficient(t)

            # Update positions based on phase
            new_members = self.members.copy()
            for i in range(self.population_size):
                # Simulate loss of contact
                if np.random.rand() < p:
                    continue  # Skip update for this member

                r1 = np.random.rand()
                r2 = np.random.rand()

                if instruction >= self.tv1:  # Exploration phase
                    new_pos = self.large_scale_search(i, r1)
                    if new_pos is None:
                        new_pos = self.raid(i, r1, w)
                    if new_pos is not None:
                        new_members[i] = new_pos
                elif self.tv2 < instruction < self.tv1:  # Transition phase
                    new_members[i] = self.transition_phase(i, r2, w, instruction)
                else:  # Exploitation phase (instruction <= tv2)
                    new_members[i] = self.arrest_rescue(t)

            self.members = new_members

            self.history.append((t, self.best_solution.copy(), self.best_value))
            print(f"Iteration {t + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
