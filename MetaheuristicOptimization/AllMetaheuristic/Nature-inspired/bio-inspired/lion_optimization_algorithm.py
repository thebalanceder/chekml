import numpy as np

class LionOptimizationAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100,
                 nomad_ratio=0.2, pride_size=5, female_ratio=0.8, roaming_ratio=0.2,
                 mating_ratio=0.2, mutation_prob=0.1, immigration_ratio=0.1):
        """
        Initialize the Lion Optimization Algorithm (LOA).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: List of tuples [(low, high), ...] for each dimension.
        - population_size: Total number of lions (solutions).
        - max_iter: Maximum number of iterations.
        - nomad_ratio: Percentage of nomad lions (%N).
        - pride_size: Number of prides (P).
        - female_ratio: Percentage of females in prides (%S).
        - roaming_ratio: Percentage of territory for male roaming (%R).
        - mating_ratio: Percentage of females that mate (%Ma).
        - mutation_prob: Mutation probability for offspring (%Mu).
        - immigration_ratio: Ratio for female immigration between prides.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Ensure bounds is a numpy array
        if self.bounds.shape != (dim, 2):
            raise ValueError(f"Bounds must be a list of {dim} tuples, each containing (lower, upper) limits.")
        self.population_size = population_size
        self.max_iter = max_iter
        self.nomad_ratio = nomad_ratio
        self.pride_size = pride_size
        self.female_ratio = female_ratio
        self.roaming_ratio = roaming_ratio
        self.mating_ratio = mating_ratio
        self.mutation_prob = mutation_prob
        self.immigration_ratio = immigration_ratio

        self.lions = None  # Population of lions
        self.best_positions = None  # Best visited position for each lion
        self.best_fitness = None  # Fitness of best positions
        self.global_best_solution = None
        self.global_best_value = float("inf")
        self.prides = []  # List of prides (each pride is a list of lion indices)
        self.nomads = []  # List of nomad lion indices
        self.genders = None  # Array of genders (True for female, False for male)
        self.history = []

    def initialize_population(self):
        """Generate initial lion population and organize into prides and nomads."""
        self.lions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                       (self.population_size, self.dim))
        self.best_positions = self.lions.copy()
        self.best_fitness = np.array([self.objective_function(lion) for lion in self.lions])

        # Assign nomads and prides
        num_nomads = int(self.nomad_ratio * self.population_size)
        indices = np.arange(self.population_size)
        np.random.shuffle(indices)
        self.nomads = indices[:num_nomads].tolist()
        resident_indices = indices[num_nomads:]

        # Divide residents into prides
        self.prides = [resident_indices[i:i + self.pride_size].tolist()
                       for i in range(0, len(resident_indices), self.pride_size)]

        # Assign genders
        self.genders = np.zeros(self.population_size, dtype=bool)  # False = male
        for pride in self.prides:
            num_females = int(self.female_ratio * len(pride))
            female_indices = np.random.choice(pride, num_females, replace=False)
            self.genders[female_indices] = True
        # Nomads have inverse gender ratio
        num_nomad_females = int((1 - self.female_ratio) * len(self.nomads))
        nomad_female_indices = np.random.choice(self.nomads, num_nomad_females, replace=False)
        self.genders[nomad_female_indices] = True

    def hunting(self, pride_indices):
        """Simulate cooperative hunting in a pride."""
        # Select hunters (female lions)
        females = [i for i in pride_indices if self.genders[i]]
        if not females:
            return
        num_hunters = max(1, len(females) // 2)
        hunters = np.random.choice(females, num_hunters, replace=False)

        # Divide hunters into three groups
        hunter_fitness = np.array([self.best_fitness[i] for i in hunters])
        group_sizes = [len(hunters) // 3] * 3
        group_sizes[0] += len(hunters) % 3  # Ensure all hunters are assigned
        groups = []
        sorted_hunters = hunters[np.argsort(hunter_fitness)]
        start = 0
        for size in group_sizes:
            groups.append(sorted_hunters[start:start + size])
            start += size
        center_group = groups[np.argmax([np.sum([self.best_fitness[i] for i in g]) for g in groups])]
        wing_groups = [g for g in groups if not np.array_equal(g, center_group)]

        # Compute dummy prey position
        prey = np.mean([self.best_positions[i] for i in hunters], axis=0)

        for hunter_idx in hunters:
            current_pos = self.lions[hunter_idx].copy()
            # Determine group
            if hunter_idx in center_group:
                # Center hunters move randomly between their position and prey
                if np.all(current_pos < prey):
                    new_pos = np.random.uniform(current_pos, prey)
                else:
                    new_pos = np.random.uniform(prey, current_pos)
            else:
                # Wing hunters use opposition-based learning
                new_pos = self.bounds[:, 0] + self.bounds[:, 1] - current_pos

            new_fitness = self.objective_function(new_pos)
            old_fitness = self.objective_function(current_pos)
            if new_fitness < old_fitness:
                self.lions[hunter_idx] = new_pos
                if new_fitness < self.best_fitness[hunter_idx]:
                    self.best_positions[hunter_idx] = new_pos
                    self.best_fitness[hunter_idx] = new_fitness
                # Prey escapes
                pi = (old_fitness - new_fitness) / old_fitness if old_fitness != 0 else 1
                prey = prey + np.random.rand() * pi * (prey - new_pos)

    def move_to_safe_place(self, pride_indices):
        """Move non-hunting females toward pride territory."""
        females = [i for i in pride_indices if self.genders[i]]
        if not females:  # Skip if no females in the pride
            return

        hunters = set(np.random.choice(females, max(1, len(females) // 2), replace=False))
        non_hunters = [i for i in females if i not in hunters]

        if not non_hunters:  # Skip if no non-hunters
            return

        # Calculate tournament size based on success
        success = np.array([1 if self.best_fitness[i] < self.objective_function(self.lions[i])
                           else 0 for i in pride_indices])
        k = np.sum(success)
        tournament_size = max(2, int(np.ceil(k / 2)))

        for idx in non_hunters:
            # Tournament selection
            candidates = np.random.choice(pride_indices, tournament_size, replace=False)
            selected_idx = min(candidates, key=lambda i: self.best_fitness[i])
            selected_pos = self.best_positions[selected_idx]

            # Move female
            current_pos = self.lions[idx]
            d = np.linalg.norm(selected_pos - current_pos)
            r1 = selected_pos - current_pos
            r1 = r1 / np.linalg.norm(r1) if np.linalg.norm(r1) != 0 else np.zeros(self.dim)
            r2 = np.random.randn(self.dim)
            r2 = r2 - np.dot(r2, r1) * r1  # Orthogonal to r1
            r2 = r2 / np.linalg.norm(r2) if np.linalg.norm(r2) != 0 else np.zeros(self.dim)
            theta = np.random.uniform(-np.pi/2, np.pi/2)
            new_pos = current_pos + 2 * d * np.random.rand() * r1 + np.random.uniform(-1, 1) * np.tan(theta) * d * r2
            new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])

            new_fitness = self.objective_function(new_pos)
            if new_fitness < self.best_fitness[idx]:
                self.lions[idx] = new_pos
                self.best_positions[idx] = new_pos
                self.best_fitness[idx] = new_fitness

    def roaming(self, pride_indices):
        """Simulate male lions roaming in pride territory."""
        males = [i for i in pride_indices if not self.genders[i]]
        for idx in males:
            territory = [self.best_positions[i] for i in pride_indices]
            num_visits = int(self.roaming_ratio * len(territory))
            visit_indices = np.random.choice(len(territory), num_visits, replace=False)
            for j in visit_indices:
                target = territory[j]
                d = np.linalg.norm(target - self.lions[idx])
                direction = target - self.lions[idx]
                direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) != 0 else np.zeros(self.dim)
                theta = np.random.uniform(-np.pi/6, np.pi/6)
                x = np.random.uniform(0, 2 * d)
                rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                if self.dim >= 2:
                    direction[:2] = rotation @ direction[:2]
                new_pos = self.lions[idx] + x * direction
                new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])
                new_fitness = self.objective_function(new_pos)
                if new_fitness < self.best_fitness[idx]:
                    self.lions[idx] = new_pos
                    self.best_positions[idx] = new_pos
                    self.best_fitness[idx] = new_fitness

    def nomad_movement(self):
        """Simulate random movement of nomad lions."""
        for idx in self.nomads:
            pr = (self.objective_function(self.lions[idx]) - min(self.best_fitness)) / \
                 (max(self.best_fitness) - min(self.best_fitness) + 1e-10)
            new_pos = self.lions[idx].copy()
            for j in range(self.dim):
                if np.random.rand() > pr:
                    new_pos[j] = np.random.uniform(self.bounds[j, 0], self.bounds[j, 1])
            new_fitness = self.objective_function(new_pos)
            if new_fitness < self.best_fitness[idx]:
                self.lions[idx] = new_pos
                self.best_positions[idx] = new_pos
                self.best_fitness[idx] = new_fitness

    def mating(self, pride_indices):
        """Simulate mating to produce offspring."""
        females = [i for i in pride_indices if self.genders[i]]
        num_mating = int(self.mating_ratio * len(females))
        mating_females = np.random.choice(females, num_mating, replace=False)
        males = [i for i in pride_indices if not self.genders[i]]

        for female_idx in mating_females:
            if not males:
                continue
            num_mates = np.random.randint(1, len(males) + 1)
            selected_males = np.random.choice(males, num_mates, replace=False)
            beta = np.random.normal(0.5, 0.1)
            s = np.zeros(len(males))
            s[[males.index(m) for m in selected_males]] = 1

            # Produce two offspring
            offspring1 = beta * self.lions[female_idx] + \
                         (1 - beta) * np.sum([self.lions[m] * s[i] for i, m in enumerate(males)], axis=0) / (sum(s) + 1e-10)
            offspring2 = (1 - beta) * self.lions[female_idx] + \
                         beta * np.sum([self.lions[m] * s[i] for i, m in enumerate(males)], axis=0) / (sum(s) + 1e-10)

            # Apply mutation
            for offspring in [offspring1, offspring2]:
                for j in range(self.dim):
                    if np.random.rand() < self.mutation_prob:
                        offspring[j] = np.random.uniform(self.bounds[j, 0], self.bounds[j, 1])

            offspring1 = np.clip(offspring1, self.bounds[:, 0], self.bounds[:, 1])
            offspring2 = np.clip(offspring2, self.bounds[:, 0], self.bounds[:, 1])

            # Add offspring to population
            if len(self.lions) < self.population_size + 2:
                self.lions = np.vstack([self.lions, offspring1, offspring2])
                self.best_positions = np.vstack([self.best_positions, offspring1, offspring2])
                self.best_fitness = np.append(self.best_fitness, [self.objective_function(offspring1),
                                                                 self.objective_function(offspring2)])
                self.genders = np.append(self.genders, [True, False])  # Randomly assign genders
                pride_indices.extend([len(self.lions) - 2, len(self.lions) - 1])

    def defense(self, pride_indices):
        """Simulate defense against mature males and nomad invasions."""
        males = [i for i in pride_indices if not self.genders[i]]
        if len(males) <= 1:
            return

        # Defense against new mature males
        male_fitness = [(i, self.best_fitness[i]) for i in males]
        male_fitness.sort(key=lambda x: x[1])
        weakest_male = male_fitness[-1][0]
        if weakest_male in pride_indices:  # Ensure weakest male is still in pride
            self.nomads.append(weakest_male)
            pride_indices.remove(weakest_male)

        # Defense against nomad males
        nomad_males = [i for i in self.nomads if not self.genders[i]]
        # Create a copy to avoid modifying nomads during iteration
        for nomad_idx in nomad_males[:]:
            if np.random.rand() < 0.5:  # Probability of invasion
                # Create a copy of males to iterate safely
                for resident_idx in males[:]:
                    if resident_idx in pride_indices and nomad_idx in self.nomads:
                        if self.best_fitness[nomad_idx] < self.best_fitness[resident_idx]:
                            self.nomads.remove(nomad_idx)
                            pride_indices.append(nomad_idx)
                            self.nomads.append(resident_idx)
                            if resident_idx in pride_indices:
                                pride_indices.remove(resident_idx)
                            break

    def immigration(self):
        """Simulate female immigration between prides or to nomads."""
        for i, pride in enumerate(self.prides):
            females = [idx for idx in pride if self.genders[idx]]
            num_immigrants = int(self.immigration_ratio * len(females))
            immigrants = np.random.choice(females, num_immigrants, replace=False)
            for idx in immigrants:
                if np.random.rand() < 0.5:  # Move to another pride or become nomad
                    pride.remove(idx)
                    if np.random.rand() < 0.5 and len(self.prides) > 1:
                        other_pride_idx = np.random.randint(0, len(self.prides))
                        while other_pride_idx == i:  # Ensure different pride
                            other_pride_idx = np.random.randint(0, len(self.prides))
                        self.prides[other_pride_idx].append(idx)
                    else:
                        self.nomads.append(idx)
        # Nomad females joining prides
        nomad_females = [idx for idx in self.nomads if self.genders[idx]]
        for idx in nomad_females:
            if np.random.rand() < 0.1 and self.prides:  # Probability of joining a pride
                self.nomads.remove(idx)
                random_pride_idx = np.random.randint(0, len(self.prides))
                self.prides[random_pride_idx].append(idx)

    def population_control(self):
        """Eliminate weakest lions to maintain population size."""
        if len(self.lions) > self.population_size:
            fitness = self.best_fitness
            worst_indices = np.argsort(fitness)[-(len(self.lions) - self.population_size):]
            mask = np.ones(len(self.lions), dtype=bool)
            mask[worst_indices] = False
            self.lions = self.lions[mask]
            self.best_positions = self.best_positions[mask]
            self.best_fitness = self.best_fitness[mask]
            self.genders = self.genders[mask]

            # Update prides and nomads
            new_nomads = []
            for idx in self.nomads:
                if idx < len(self.lions):
                    new_nomads.append(idx)
            self.nomads = new_nomads
            new_prides = []
            for pride in self.prides:
                new_pride = [idx for idx in pride if idx < len(self.lions)]
                if new_pride:  # Only keep non-empty prides
                    new_prides.append(new_pride)
            self.prides = new_prides

    def optimize(self):
        """Run the Lion Optimization Algorithm."""
        self.initialize_population()
        for generation in range(self.max_iter):
            # Update global best
            min_idx = np.argmin(self.best_fitness)
            if self.best_fitness[min_idx] < self.global_best_value:
                self.global_best_solution = self.best_positions[min_idx].copy()
                self.global_best_value = self.best_fitness[min_idx]

            # Pride operations
            for pride in self.prides[:]:  # Copy to avoid modification issues
                self.hunting(pride)
                self.move_to_safe_place(pride)
                self.roaming(pride)
                self.mating(pride)
                self.defense(pride)

            # Nomad operations
            self.nomad_movement()

            # Immigration
            self.immigration()

            # Population control
            self.population_control()

            self.history.append((generation, self.global_best_solution.copy(), self.global_best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.global_best_value}")

        return self.global_best_solution, self.global_best_value, self.history
