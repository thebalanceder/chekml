# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, cos, sin, sqrt
from libc.stdlib cimport rand, RAND_MAX

np.import_array()

# Define numpy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class MayflyOptimizer:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        double inertia_weight
        double inertia_damp
        double personal_coeff
        double global_coeff1
        double global_coeff2
        double distance_coeff
        double nuptial_dance
        double random_flight
        double dance_damp
        double flight_damp
        int num_offspring
        int num_mutants
        double mutation_rate
        np.ndarray vel_max
        np.ndarray vel_min
        dict males
        dict females
        dict global_best
        list best_solution_history
        int function_evaluations

    def __init__(self, objective_function, int dim, bounds, int population_size=20, int max_iter=100,
                 double inertia_weight=0.8, double inertia_damp=1.0, double personal_coeff=1.0,
                 double global_coeff1=1.5, double global_coeff2=1.5, double distance_coeff=2.0,
                 double nuptial_dance=5.0, double random_flight=1.0, double dance_damp=0.8,
                 double flight_damp=0.99, int num_offspring=20, int num_mutants=1, double mutation_rate=0.01):
        """
        Initialize the Mayfly Optimization Algorithm.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_populations(self):
        """ Initialize male and female mayfly populations """
        cdef np.ndarray[DTYPE_t, ndim=2] male_positions = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim)).astype(DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] male_velocities = np.zeros((self.population_size, self.dim), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] male_costs = np.zeros(self.population_size, dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] male_best_positions = male_positions.copy()
        cdef np.ndarray[DTYPE_t, ndim=1] male_best_costs = np.zeros(self.population_size, dtype=DTYPE)

        cdef np.ndarray[DTYPE_t, ndim=2] female_positions = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim)).astype(DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] female_velocities = np.zeros((self.population_size, self.dim), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] female_costs = np.zeros(self.population_size, dtype=DTYPE)

        self.males = {
            'positions': male_positions,
            'velocities': male_velocities,
            'costs': male_costs,
            'best_positions': male_best_positions,
            'best_costs': male_best_costs
        }
        self.females = {
            'positions': female_positions,
            'velocities': female_velocities,
            'costs': female_costs
        }

        # Evaluate initial populations
        cdef int i
        for i in range(self.population_size):
            male_costs[i] = self.objective_function(male_positions[i])
            male_best_costs[i] = male_costs[i]
            self.function_evaluations += 1

            if male_best_costs[i] < self.global_best['cost']:
                self.global_best['position'] = male_positions[i].copy()
                self.global_best['cost'] = male_best_costs[i]

            female_costs[i] = self.objective_function(female_positions[i])
            self.function_evaluations += 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def crossover(self, np.ndarray[DTYPE_t, ndim=1] male_pos, np.ndarray[DTYPE_t, ndim=1] female_pos):
        """ Perform crossover between male and female positions """
        cdef np.ndarray[DTYPE_t, ndim=1] L = np.random.uniform(0, 1, self.dim).astype(DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] off1 = L * male_pos + (1 - L) * female_pos
        cdef np.ndarray[DTYPE_t, ndim=1] off2 = L * female_pos + (1 - L) * male_pos
        off1 = np.clip(off1, self.bounds[:, 0], self.bounds[:, 1])
        off2 = np.clip(off2, self.bounds[:, 0], self.bounds[:, 1])
        return off1, off2

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def mutate(self, np.ndarray[DTYPE_t, ndim=1] position):
        """ Apply mutation to a position """
        cdef int n_var = self.dim
        cdef int n_mu = int(np.ceil(self.mutation_rate * n_var))
        cdef np.ndarray[DTYPE_t, ndim=1] sigma = 0.1 * (self.bounds[:, 1] - self.bounds[:, 0])
        cdef np.ndarray[DTYPE_t, ndim=1] mutated = position.copy()
        cdef np.ndarray[np.int32_t, ndim=1] j = np.random.choice(n_var, n_mu, replace=False).astype(np.int32)
        cdef int k
        for k in range(n_mu):
            mutated[j[k]] = position[j[k]] + sigma[j[k]] * np.random.randn()
        mutated = np.clip(mutated, self.bounds[:, 0], self.bounds[:, 1])
        return mutated

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_females(self):
        """ Update female mayflies' velocities and positions """
        cdef np.ndarray[DTYPE_t, ndim=2] male_positions = self.males['positions']
        cdef np.ndarray[DTYPE_t, ndim=2] female_positions = self.females['positions']
        cdef np.ndarray[DTYPE_t, ndim=2] female_velocities = self.females['velocities']
        cdef np.ndarray[DTYPE_t, ndim=1] male_costs = self.males['costs']
        cdef np.ndarray[DTYPE_t, ndim=1] female_costs = self.females['costs']
        cdef np.ndarray[DTYPE_t, ndim=1] rmf, e
        cdef int i, j
        for i in range(self.population_size):
            rmf = male_positions[i] - female_positions[i]
            e = np.random.uniform(-1, 1, self.dim).astype(DTYPE)
            if female_costs[i] > male_costs[i]:
                for j in range(self.dim):
                    female_velocities[i, j] = (self.inertia_weight * female_velocities[i, j] +
                        self.global_coeff2 * exp(-self.distance_coeff * rmf[j]**2) * rmf[j])
            else:
                for j in range(self.dim):
                    female_velocities[i, j] = (self.inertia_weight * female_velocities[i, j] +
                        self.random_flight * e[j])

            # Apply velocity limits
            for j in range(self.dim):
                if female_velocities[i, j] < self.vel_min[j]:
                    female_velocities[i, j] = self.vel_min[j]
                elif female_velocities[i, j] > self.vel_max[j]:
                    female_velocities[i, j] = self.vel_max[j]

            # Update position
            for j in range(self.dim):
                female_positions[i, j] += female_velocities[i, j]
                if female_positions[i, j] < self.bounds[j, 0]:
                    female_positions[i, j] = self.bounds[j, 0]
                elif female_positions[i, j] > self.bounds[j, 1]:
                    female_positions[i, j] = self.bounds[j, 1]

            # Evaluate new position
            female_costs[i] = self.objective_function(female_positions[i])
            self.function_evaluations += 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_males(self):
        """ Update male mayflies' velocities and positions """
        cdef np.ndarray[DTYPE_t, ndim=2] positions = self.males['positions']
        cdef np.ndarray[DTYPE_t, ndim=2] velocities = self.males['velocities']
        cdef np.ndarray[DTYPE_t, ndim=1] costs = self.males['costs']
        cdef np.ndarray[DTYPE_t, ndim=2] best_positions = self.males['best_positions']
        cdef np.ndarray[DTYPE_t, ndim=1] best_costs = self.males['best_costs']
        cdef np.ndarray[DTYPE_t, ndim=1] global_best_pos = self.global_best['position']
        cdef double global_best_cost = self.global_best['cost']
        cdef np.ndarray[DTYPE_t, ndim=1] rpbest, rgbest, e
        cdef int i, j
        for i in range(self.population_size):
            rpbest = best_positions[i] - positions[i]
            rgbest = global_best_pos - positions[i]
            e = np.random.uniform(-1, 1, self.dim).astype(DTYPE)

            if costs[i] > global_best_cost:
                for j in range(self.dim):
                    velocities[i, j] = (self.inertia_weight * velocities[i, j] +
                        self.personal_coeff * exp(-self.distance_coeff * rpbest[j]**2) * rpbest[j] +
                        self.global_coeff1 * exp(-self.distance_coeff * rgbest[j]**2) * rgbest[j])
            else:
                for j in range(self.dim):
                    velocities[i, j] = (self.inertia_weight * velocities[i, j] +
                        self.nuptial_dance * e[j])

            # Apply velocity limits
            for j in range(self.dim):
                if velocities[i, j] < self.vel_min[j]:
                    velocities[i, j] = self.vel_min[j]
                elif velocities[i, j] > self.vel_max[j]:
                    velocities[i, j] = self.vel_max[j]

            # Update position
            for j in range(self.dim):
                positions[i, j] += velocities[i, j]
                if positions[i, j] < self.bounds[j, 0]:
                    positions[i, j] = self.bounds[j, 0]
                elif positions[i, j] > self.bounds[j, 1]:
                    positions[i, j] = self.bounds[j, 1]

            # Evaluate new position
            costs[i] = self.objective_function(positions[i])
            self.function_evaluations += 1

            # Update personal best
            if costs[i] < best_costs[i]:
                for j in range(self.dim):
                    best_positions[i, j] = positions[i, j]
                best_costs[i] = costs[i]
                if best_costs[i] < global_best_cost:
                    self.global_best['position'] = positions[i].copy()
                    self.global_best['cost'] = best_costs[i]

    def mating_phase(self):
        """ Perform mating to generate offspring """
        offspring = []
        cdef int k
        for k in range(self.num_offspring // 2):
            p1 = self.males['positions'][k]
            p2 = self.females['positions'][k]
            off1_pos, off2_pos = self.crossover(p1, p2)
            
            off1 = {
                'position': off1_pos,
                'cost': self.objective_function(off1_pos),
                'velocity': np.zeros(self.dim, dtype=DTYPE),
                'best_position': off1_pos.copy(),
                'best_cost': self.objective_function(off1_pos)
            }
            off2 = {
                'position': off2_pos,
                'cost': self.objective_function(off2_pos),
                'velocity': np.zeros(self.dim, dtype=DTYPE),
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

    def mutation_phase(self, list offspring):
        """ Apply mutation to offspring """
        mutants = []
        cdef int i
        for _ in range(self.num_mutants):
            i = np.random.randint(len(offspring))
            mutated_pos = self.mutate(offspring[i]['position'])
            mutant = {
                'position': mutated_pos,
                'cost': self.objective_function(mutated_pos),
                'velocity': np.zeros(self.dim, dtype=DTYPE),
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
        cdef int iteration, i, j, split
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
            self.males['positions'] = np.array([male_positions[i] for i in male_sort], dtype=DTYPE)
            self.males['velocities'] = np.array([male_velocities[i] for i in male_sort], dtype=DTYPE)
            self.males['costs'] = np.array([male_costs[i] for i in male_sort], dtype=DTYPE)
            self.males['best_positions'] = np.array([male_best_positions[i] for i in male_sort], dtype=DTYPE)
            self.males['best_costs'] = np.array([male_best_costs[i] for i in male_sort], dtype=DTYPE)

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
            self.females['positions'] = np.array([female_positions[i] for i in female_sort], dtype=DTYPE)
            self.females['velocities'] = np.array([female_velocities[i] for i in female_sort], dtype=DTYPE)
            self.females['costs'] = np.array([female_costs[i] for i in female_sort], dtype=DTYPE)

            # Update parameters
            self.inertia_weight *= self.inertia_damp
            self.nuptial_dance *= self.dance_damp
            self.random_flight *= self.flight_damp

            # Store best solution
            self.best_solution_history.append(self.global_best['cost'])
            print(f"Iteration {iteration + 1}: Evaluations = {self.function_evaluations}, "
                  f"Best Cost = {self.global_best['cost']}")

        return self.global_best['position'], self.global_best['cost'], self.best_solution_history

# Example objective functions
def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    n = len(x)
    A = 10
    return n * A + np.sum(x**2 - A * cos(2 * np.pi * x))

# Example usage
if __name__ == "__main__":
    dim = 50
    bounds = [(-10, 10)] * dim
    optimizer = MayflyOptimizer(sphere, dim, bounds)
    best_position, best_cost, history = optimizer.optimize()
    print(f"Best Position: {best_position}")
    print(f"Best Cost: {best_cost}")
