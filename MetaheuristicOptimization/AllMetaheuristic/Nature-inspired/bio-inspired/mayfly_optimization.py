import numpy as np

class MayflyOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=20, max_iter=100, 
                 inertia_weight=0.8, inertia_damp=1.0, personal_coeff=1.0, 
                 global_coeff1=1.5, global_coeff2=1.5, distance_coeff=2.0, 
                 nuptial_dance=5.0, random_flight=1.0, dance_damp=0.8, 
                 flight_damp=0.99, num_offspring=20, num_mutants=1, mutation_rate=0.01):
        """
        Initialize the Mayfly Optimization Algorithm.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of mayflies (males and females).
        - max_iter: Maximum number of iterations.
        - inertia_weight: Inertia weight for velocity update.
        - inertia_damp: Inertia weight damping ratio.
        - personal_coeff: Personal learning coefficient.
        - global_coeff1, global_coeff2: Global learning coefficients.
        - distance_coeff: Distance sight coefficient.
        - nuptial_dance: Nuptial dance coefficient.
        - random_flight: Random flight coefficient.
        - dance_damp: Nuptial dance damping ratio.
        - flight_damp: Random flight damping ratio.
        - num_offspring: Number of offspring (also parents).
        - num_mutants: Number of mutants.
        - mutation_rate: Mutation rate.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.inertia_weight = inertia_weight
        self.inertia_damp = inertia_damp
        self.personal_coeff = personal_coeff
        self.global_coeff1 = global_coeff1
        self.global_coeff2 = global_coeff2
        self.distance_coeff = distance_coeff
        self.nuptial_dance = nuptial_dance
        self.random_flight = random_flight
        self.dance_damp = dance_damp
        self.flight_damp = flight_damp
        self.num_offspring = num_offspring
        self.num_mutants = num_mutants
        self.mutation_rate = mutation_rate

        # Velocity limits
        self.vel_max = 0.1 * (self.bounds[:, 1] - self.bounds[:, 0])
        self.vel_min = -self.vel_max

        # Initialize populations
        self.males = None
        self.females = None
        self.global_best = {'position': None, 'cost': float('inf')}
        self.best_solution_history = []
        self.function_evaluations = 0

    def initialize_populations(self):
        """ Initialize male and female mayfly populations """
        self.males = {
            'positions': np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                         (self.population_size, self.dim)),
            'velocities': np.zeros((self.population_size, self.dim)),
            'costs': np.zeros(self.population_size),
            'best_positions': np.zeros((self.population_size, self.dim)),
            'best_costs': np.zeros(self.population_size)
        }
        self.females = {
            'positions': np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                         (self.population_size, self.dim)),
            'velocities': np.zeros((self.population_size, self.dim)),
            'costs': np.zeros(self.population_size)
        }

        # Evaluate initial populations
        for i in range(self.population_size):
            self.males['costs'][i] = self.objective_function(self.males['positions'][i])
            self.males['best_positions'][i] = self.males['positions'][i].copy()
            self.males['best_costs'][i] = self.males['costs'][i]
            self.function_evaluations += 1

            if self.males['best_costs'][i] < self.global_best['cost']:
                self.global_best['position'] = self.males['best_positions'][i].copy()
                self.global_best['cost'] = self.males['best_costs'][i]

            self.females['costs'][i] = self.objective_function(self.females['positions'][i])
            self.function_evaluations += 1

    def crossover(self, male_pos, female_pos):
        """ Perform crossover between male and female positions """
        L = np.random.uniform(0, 1, self.dim)
        off1 = L * male_pos + (1 - L) * female_pos
        off2 = L * female_pos + (1 - L) * male_pos
        off1 = np.clip(off1, self.bounds[:, 0], self.bounds[:, 1])
        off2 = np.clip(off2, self.bounds[:, 0], self.bounds[:, 1])
        return off1, off2

    def mutate(self, position):
        """ Apply mutation to a position """
        n_var = self.dim
        n_mu = int(np.ceil(self.mutation_rate * n_var))
        j = np.random.choice(n_var, n_mu, replace=False)
        sigma = 0.1 * (self.bounds[:, 1] - self.bounds[:, 0])
        mutated = position.copy()
        mutated[j] = position[j] + sigma[j] * np.random.randn(len(j))
        mutated = np.clip(mutated, self.bounds[:, 0], self.bounds[:, 1])
        return mutated

    def update_females(self):
        """ Update female mayflies' velocities and positions """
        for i in range(self.population_size):
            rmf = self.males['positions'][i] - self.females['positions'][i]
            e = np.random.uniform(-1, 1, self.dim)
            if self.females['costs'][i] > self.males['costs'][i]:
                self.females['velocities'][i] = (self.inertia_weight * self.females['velocities'][i] +
                    self.global_coeff2 * np.exp(-self.distance_coeff * rmf**2) * rmf)
            else:
                self.females['velocities'][i] = (self.inertia_weight * self.females['velocities'][i] +
                    self.random_flight * e)

            # Apply velocity limits
            self.females['velocities'][i] = np.clip(self.females['velocities'][i], self.vel_min, self.vel_max)

            # Update position
            self.females['positions'][i] += self.females['velocities'][i]
            self.females['positions'][i] = np.clip(self.females['positions'][i], 
                                                  self.bounds[:, 0], self.bounds[:, 1])

            # Evaluate new position
            self.females['costs'][i] = self.objective_function(self.females['positions'][i])
            self.function_evaluations += 1

    def update_males(self):
        """ Update male mayflies' velocities and positions """
        for i in range(self.population_size):
            rpbest = self.males['best_positions'][i] - self.males['positions'][i]
            rgbest = self.global_best['position'] - self.males['positions'][i]
            e = np.random.uniform(-1, 1, self.dim)

            if self.males['costs'][i] > self.global_best['cost']:
                self.males['velocities'][i] = (self.inertia_weight * self.males['velocities'][i] +
                    self.personal_coeff * np.exp(-self.distance_coeff * rpbest**2) * rpbest +
                    self.global_coeff1 * np.exp(-self.distance_coeff * rgbest**2) * rgbest)
            else:
                self.males['velocities'][i] = (self.inertia_weight * self.males['velocities'][i] +
                    self.nuptial_dance * e)

            # Apply velocity limits
            self.males['velocities'][i] = np.clip(self.males['velocities'][i], self.vel_min, self.vel_max)

            # Update position
            self.males['positions'][i] += self.males['velocities'][i]
            self.males['positions'][i] = np.clip(self.males['positions'][i], 
                                                self.bounds[:, 0], self.bounds[:, 1])

            # Evaluate new position
            self.males['costs'][i] = self.objective_function(self.males['positions'][i])
            self.function_evaluations += 1

            # Update personal best
            if self.males['costs'][i] < self.males['best_costs'][i]:
                self.males['best_positions'][i] = self.males['positions'][i].copy()
                self.males['best_costs'][i] = self.males['costs'][i]
                if self.males['best_costs'][i] < self.global_best['cost']:
                    self.global_best['position'] = self.males['best_positions'][i].copy()
                    self.global_best['cost'] = self.males['best_costs'][i]

    def mating_phase(self):
        """ Perform mating to generate offspring """
        offspring = []
        for k in range(self.num_offspring // 2):
            p1 = self.males['positions'][k]
            p2 = self.females['positions'][k]
            off1_pos, off2_pos = self.crossover(p1, p2)
            
            off1 = {
                'position': off1_pos,
                'cost': self.objective_function(off1_pos),
                'velocity': np.zeros(self.dim),
                'best_position': off1_pos.copy(),
                'best_cost': self.objective_function(off1_pos)
            }
            off2 = {
                'position': off2_pos,
                'cost': self.objective_function(off2_pos),
                'velocity': np.zeros(self.dim),
                'best_position': off2_pos.copy(),
                'best_cost': self.objective_function(off2_pos)
            }
            self.function_evaluations += 2

            if off1['cost'] < self.global_best['cost']:
                self.global_best = {'position': off1['best_position'].copy(), 'cost': off1['best_cost']}
            if off2['cost'] < self.global_best['cost']:
                self.global_best = {'position': off2['best_position'].copy(), 'cost': off2['best_cost']}

            offspring.extend([off1, off2])

        return offspring

    def mutation_phase(self, offspring):
        """ Apply mutation to offspring """
        mutants = []
        for _ in range(self.num_mutants):
            i = np.random.randint(len(offspring))
            mutated_pos = self.mutate(offspring[i]['position'])
            mutant = {
                'position': mutated_pos,
                'cost': self.objective_function(mutated_pos),
                'velocity': np.zeros(self.dim),
                'best_position': mutated_pos.copy(),
                'best_cost': self.objective_function(mutated_pos)
            }
            self.function_evaluations += 1
            if mutant['cost'] < self.global_best['cost']:
                self.global_best = {'position': mutant['best_position'].copy(), 'cost': mutant['best_cost']}
            mutants.append(mutant)
        return mutants

    def optimize(self):
        """ Run the Mayfly Optimization Algorithm """
        self.initialize_populations()
        for iteration in range(self.max_iter):
            # Update females and males
            self.update_females()
            self.update_males()

            # Sort populations by cost
            male_sort = np.argsort(self.males['costs'])
            female_sort = np.argsort(self.females['costs'])
            for key in self.males:
                self.males[key] = self.males[key][male_sort]
            for key in self.females:
                self.females[key] = self.females[key][female_sort]

            # Mating and mutation
            offspring = self.mating_phase()
            mutants = self.mutation_phase(offspring)
            offspring.extend(mutants)

            # Merge and select best individuals
            split = len(offspring) // 2
            new_males = [offspring[i] for i in range(split)]
            new_females = [offspring[i] for i in range(split, len(offspring))]

            # Update male population
            male_pop = [self.males] + new_males
            male_costs = []
            male_positions = []
            male_velocities = []
            male_best_positions = []
            male_best_costs = []
            for m in male_pop:
                if isinstance(m, dict) and 'costs' in m:  # Main population
                    male_costs.extend(m['costs'].tolist())
                    male_positions.extend(m['positions'].tolist())
                    male_velocities.extend(m['velocities'].tolist())
                    male_best_positions.extend(m['best_positions'].tolist())
                    male_best_costs.extend(m['best_costs'].tolist())
                else:  # Individual offspring/mutant
                    male_costs.append(m['cost'])
                    male_positions.append(m['position'])
                    male_velocities.append(m['velocity'])
                    male_best_positions.append(m['best_position'])
                    male_best_costs.append(m['best_cost'])

            male_sort = np.argsort(male_costs)[:self.population_size]
            self.males['positions'] = np.array([male_positions[i] for i in male_sort])
            self.males['velocities'] = np.array([male_velocities[i] for i in male_sort])
            self.males['costs'] = np.array([male_costs[i] for i in male_sort])
            self.males['best_positions'] = np.array([male_best_positions[i] for i in male_sort])
            self.males['best_costs'] = np.array([male_best_costs[i] for i in male_sort])

            # Update female population
            female_pop = [self.females] + new_females
            female_costs = []
            female_positions = []
            female_velocities = []
            for f in female_pop:
                if isinstance(f, dict) and 'costs' in f:  # Main population
                    female_costs.extend(f['costs'].tolist())
                    female_positions.extend(f['positions'].tolist())
                    female_velocities.extend(f['velocities'].tolist())
                else:  # Individual offspring/mutant
                    female_costs.append(f['cost'])
                    female_positions.append(f['position'])
                    female_velocities.append(f['velocity'])

            female_sort = np.argsort(female_costs)[:self.population_size]
            self.females['positions'] = np.array([female_positions[i] for i in female_sort])
            self.females['velocities'] = np.array([female_velocities[i] for i in female_sort])
            self.females['costs'] = np.array([female_costs[i] for i in female_sort])

            # Update parameters
            self.inertia_weight *= self.inertia_damp
            self.nuptial_dance *= self.dance_damp
            self.random_flight *= self.flight_damp

            # Store best solution
            self.best_solution_history.append(self.global_best['cost'])
            print(f"Iteration {iteration + 1}: Evaluations = {self.function_evaluations}, "
                  f"Best Cost = {self.global_best['cost']}")

        return self.global_best['position'], self.global_best['cost'], self.best_solution_history

