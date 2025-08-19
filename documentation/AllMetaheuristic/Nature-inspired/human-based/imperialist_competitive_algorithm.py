import numpy as np

class ImperialistCompetitiveAlgorithm:
    def __init__(self, objective_function, dim, bounds, num_countries=200, num_initial_imperialists=8, 
                 max_decades=2000, revolution_rate=0.3, assimilation_coeff=2.0, 
                 assimilation_angle_coeff=0.5, zeta=0.02, damp_ratio=0.99, 
                 uniting_threshold=0.02, stop_if_single_empire=False):
        """
        Initialize the Imperialist Competitive Algorithm (ICA) optimizer.

        Parameters:
        - objective_function: Function to optimize (returns cost; lower is better).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - num_countries: Total number of initial countries.
        - num_initial_imperialists: Number of initial imperialists.
        - max_decades: Maximum number of iterations (decades).
        - revolution_rate: Rate of sudden socio-political changes in colonies.
        - assimilation_coeff: Coefficient for colony movement toward imperialist.
        - assimilation_angle_coeff: Coefficient for angular assimilation (not used in this version).
        - zeta: Weight of colonies' mean cost in total empire cost.
        - damp_ratio: Damping factor for revolution rate.
        - uniting_threshold: Threshold for uniting similar empires (as % of search space).
        - stop_if_single_empire: If True, stop when only one empire remains.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.num_countries = num_countries
        self.num_imperialists = num_initial_imperialists
        self.num_colonies = num_countries - num_initial_imperialists
        self.max_decades = max_decades
        self.revolution_rate = revolution_rate
        self.assimilation_coeff = assimilation_coeff
        self.assimilation_angle_coeff = assimilation_angle_coeff
        self.zeta = zeta
        self.damp_ratio = damp_ratio
        self.uniting_threshold = uniting_threshold
        self.stop_if_single_empire = stop_if_single_empire

        self.countries = None
        self.costs = None
        self.empires = []
        self.best_solution = None
        self.best_cost = float("inf")
        self.history = []

    def generate_new_country(self, num_countries):
        """Generate random countries within bounds."""
        var_min = self.bounds[:, 0]
        var_max = self.bounds[:, 1]
        return np.random.uniform(var_min, var_max, (num_countries, self.dim))

    def create_initial_empires(self):
        """Initialize empires by assigning imperialists and colonies."""
        # Sort countries by cost (ascending, as lower cost is better)
        sort_indices = np.argsort(self.costs)
        self.costs = self.costs[sort_indices]
        self.countries = self.countries[sort_indices]

        # Assign imperialists and colonies
        imperialists_pos = self.countries[:self.num_imperialists]
        imperialists_cost = self.costs[:self.num_imperialists]
        colonies_pos = self.countries[self.num_imperialists:]
        colonies_cost = self.costs[self.num_imperialists:]

        # Compute imperialist power
        max_cost = np.max(imperialists_cost)
        if max_cost > 0:
            power = 1.3 * max_cost - imperialists_cost
        else:
            power = 0.7 * max_cost - imperialists_cost

        # Assign number of colonies to each imperialist
        num_colonies = np.round(power / np.sum(power) * self.num_colonies).astype(int)
        num_colonies[-1] = self.num_colonies - np.sum(num_colonies[:-1])  # Ensure total matches

        # Randomly assign colonies
        random_indices = np.random.permutation(self.num_colonies)
        start_idx = 0
        self.empires = []
        for i in range(self.num_imperialists):
            n_cols = num_colonies[i]
            indices = random_indices[start_idx:start_idx + n_cols]
            start_idx += n_cols

            empire = {
                'ImperialistPosition': imperialists_pos[i],
                'ImperialistCost': imperialists_cost[i],
                'ColoniesPosition': colonies_pos[indices] if n_cols > 0 else np.array([]).reshape(0, self.dim),
                'ColoniesCost': colonies_cost[indices] if n_cols > 0 else np.array([]),
                'TotalCost': None
            }
            # If no colonies, generate one
            if n_cols == 0:
                empire['ColoniesPosition'] = self.generate_new_country(1)
                empire['ColoniesCost'] = np.array([self.objective_function(empire['ColoniesPosition'][0])])
            empire['TotalCost'] = empire['ImperialistCost'] + self.zeta * np.mean(empire['ColoniesCost']) if empire['ColoniesCost'].size > 0 else empire['ImperialistCost']
            self.empires.append(empire)

    def assimilate_colonies(self, empire):
        """Move colonies toward their imperialist (assimilation policy)."""
        if empire['ColoniesPosition'].size == 0:
            return empire

        num_cols = empire['ColoniesPosition'].shape[0]
        vector = np.tile(empire['ImperialistPosition'], (num_cols, 1)) - empire['ColoniesPosition']
        empire['ColoniesPosition'] += 2 * self.assimilation_coeff * np.random.rand(num_cols, self.dim) * vector

        # Clip to bounds
        empire['ColoniesPosition'] = np.clip(empire['ColoniesPosition'], self.bounds[:, 0], self.bounds[:, 1])
        return empire

    def revolve_colonies(self, empire):
        """Introduce sudden changes in some colonies (revolution)."""
        if empire['ColoniesCost'].size == 0:
            return empire

        num_revolving = int(np.round(self.revolution_rate * empire['ColoniesCost'].size))
        if num_revolving == 0:
            return empire

        indices = np.random.choice(empire['ColoniesCost'].size, num_revolving, replace=False)
        empire['ColoniesPosition'][indices] = self.generate_new_country(num_revolving)
        return empire

    def possess_empire(self, empire):
        """Allow a colony to become the imperialist if it has a lower cost."""
        if empire['ColoniesCost'].size == 0:
            return empire

        min_colony_cost = np.min(empire['ColoniesCost'])
        best_colony_idx = np.argmin(empire['ColoniesCost'])
        if min_colony_cost < empire['ImperialistCost']:
            # Swap imperialist and best colony
            old_imp_pos = empire['ImperialistPosition'].copy()
            old_imp_cost = empire['ImperialistCost']
            empire['ImperialistPosition'] = empire['ColoniesPosition'][best_colony_idx]
            empire['ImperialistCost'] = empire['ColoniesCost'][best_colony_idx]
            empire['ColoniesPosition'][best_colony_idx] = old_imp_pos
            empire['ColoniesCost'][best_colony_idx] = old_imp_cost
        return empire

    def unite_similar_empires(self):
        """Merge empires that are too close."""
        threshold = self.uniting_threshold * np.linalg.norm(self.bounds[:, 1] - self.bounds[:, 0])
        i = 0
        while i < len(self.empires) - 1:
            j = i + 1
            while j < len(self.empires):
                distance = np.linalg.norm(self.empires[i]['ImperialistPosition'] - self.empires[j]['ImperialistPosition'])
                if distance <= threshold:
                    # Merge empires
                    if self.empires[i]['ImperialistCost'] < self.empires[j]['ImperialistCost']:
                        better_idx, worse_idx = i, j
                    else:
                        better_idx, worse_idx = j, i

                    # Combine colonies
                    self.empires[better_idx]['ColoniesPosition'] = np.vstack((
                        self.empires[better_idx]['ColoniesPosition'],
                        self.empires[worse_idx]['ImperialistPosition'].reshape(1, -1),
                        self.empires[worse_idx]['ColoniesPosition']
                    ))
                    self.empires[better_idx]['ColoniesCost'] = np.concatenate((
                        self.empires[better_idx]['ColoniesCost'],
                        [self.empires[worse_idx]['ImperialistCost']],
                        self.empires[worse_idx]['ColoniesCost']
                    ))
                    # Update total cost
                    self.empires[better_idx]['TotalCost'] = self.empires[better_idx]['ImperialistCost'] + \
                        self.zeta * np.mean(self.empires[better_idx]['ColoniesCost']) if self.empires[better_idx]['ColoniesCost'].size > 0 else \
                        self.empires[better_idx]['ImperialistCost']
                    # Remove worse empire
                    self.empires.pop(worse_idx)
                    continue
                j += 1
            i += 1

    def imperialistic_competition(self):
        """Perform competition among empires, transferring colonies from weakest to others."""
        if np.random.rand() > 0.11 or len(self.empires) <= 1:
            return

        total_costs = np.array([empire['TotalCost'] for empire in self.empires])
        max_cost = np.max(total_costs)
        powers = max_cost - total_costs
        if np.sum(powers) == 0:
            return
        possession_prob = powers / np.sum(powers)
        selected_idx = np.argmax(possession_prob - np.random.rand(len(possession_prob)))
        weakest_idx = np.argmax(total_costs)

        if self.empires[weakest_idx]['ColoniesCost'].size == 0:
            return

        # Transfer a random colony
        colony_idx = np.random.randint(0, self.empires[weakest_idx]['ColoniesCost'].size)
        self.empires[selected_idx]['ColoniesPosition'] = np.vstack((
            self.empires[selected_idx]['ColoniesPosition'],
            self.empires[weakest_idx]['ColoniesPosition'][colony_idx].reshape(1, -1)
        ))
        self.empires[selected_idx]['ColoniesCost'] = np.append(
            self.empires[selected_idx]['ColoniesCost'],
            self.empires[weakest_idx]['ColoniesCost'][colony_idx]
        )
        # Remove colony from weakest empire
        self.empires[weakest_idx]['ColoniesPosition'] = np.delete(
            self.empires[weakest_idx]['ColoniesPosition'], colony_idx, axis=0
        )
        self.empires[weakest_idx]['ColoniesCost'] = np.delete(
            self.empires[weakest_idx]['ColoniesCost'], colony_idx
        )
        # Collapse weakest empire if it has no colonies
        if self.empires[weakest_idx]['ColoniesCost'].size <= 1:
            self.empires[selected_idx]['ColoniesPosition'] = np.vstack((
                self.empires[selected_idx]['ColoniesPosition'],
                self.empires[weakest_idx]['ImperialistPosition'].reshape(1, -1)
            ))
            self.empires[selected_idx]['ColoniesCost'] = np.append(
                self.empires[selected_idx]['ColoniesCost'],
                self.empires[weakest_idx]['ImperialistCost']
            )
            self.empires.pop(weakest_idx)

    def optimize(self):
        """Run the Imperialist Competitive Algorithm."""
        # Initialize countries
        self.countries = self.generate_new_country(self.num_countries)
        self.costs = np.array([self.objective_function(country) for country in self.countries])
        self.create_initial_empires()

        for decade in range(self.max_decades):
            self.revolution_rate *= self.damp_ratio
            for i in range(len(self.empires)):
                # Assimilation
                self.empires[i] = self.assimilate_colonies(self.empires[i])
                # Revolution
                self.empires[i] = self.revolve_colonies(self.empires[i])
                # Update colony costs
                if self.empires[i]['ColoniesPosition'].size > 0:
                    self.empires[i]['ColoniesCost'] = np.array([
                        self.objective_function(pos) for pos in self.empires[i]['ColoniesPosition']
                    ])
                # Possession
                self.empires[i] = self.possess_empire(self.empires[i])
                # Update total cost
                self.empires[i]['TotalCost'] = self.empires[i]['ImperialistCost'] + \
                    self.zeta * np.mean(self.empires[i]['ColoniesCost']) if self.empires[i]['ColoniesCost'].size > 0 else \
                    self.empires[i]['ImperialistCost']

            # Unite similar empires
            self.unite_similar_empires()
            # Imperialistic competition
            self.imperialistic_competition()

            # Track best solution
            imperialists_costs = np.array([empire['ImperialistCost'] for empire in self.empires])
            min_idx = np.argmin(imperialists_costs)
            if imperialists_costs[min_idx] < self.best_cost:
                self.best_solution = self.empires[min_idx]['ImperialistPosition'].copy()
                self.best_cost = imperialists_costs[min_idx]

            self.history.append((decade, self.best_solution.copy(), self.best_cost))
            print(f"Decade {decade + 1}: Best Cost = {self.best_cost}")

            if len(self.empires) == 1 and self.stop_if_single_empire:
                break

        return self.best_solution, self.best_cost, self.history
