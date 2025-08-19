import numpy as np

class KeshtelOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=30, max_iter=100,
                 s_max=4, p1=0.2, p2=0.5):
        """
        Keshtel Algorithm (KA)

        Parameters:
        - objective_function: Function to minimize.
        - dim: Number of dimensions.
        - bounds: Tuple of (lower_bound, upper_bound) for each dimension.
        - population_size: Total number of agents (keshtels).
        - max_iter: Maximum iterations.
        - s_max: Maximum swirl strength.
        - p1: Ratio of best population (N1).
        - p2: Ratio of middle population (N2).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.max_iter = max_iter
        self.s_max = s_max
        self.p1 = p1
        self.p2 = p2

        self.population = []
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def random_solution(self):
        """ Generate a random solution within bounds """
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)

    def initialize_population(self):
        """ Initialize all keshtel positions and evaluate their cost """
        self.population = [{
            'position': self.random_solution(),
            'cost': None,
            'nn': None
        } for _ in range(self.population_size)]

        for ind in self.population:
            ind['cost'] = self.objective_function(ind['position'])

        self.population.sort(key=lambda x: x['cost'])

    def nearest_neighbor(self, target_position, population):
        """ Find the nearest neighbor to the target_position in the given population """
        distances = [np.linalg.norm(target_position - p['position']) for p in population]
        return population[np.argmin(distances)]

    def swirl(self, individual, neighbor, s):
        """ Swirl behavior based on current strength S and max strength Smax """
        pos = individual['position']
        nn_pos = neighbor['position']
        swirl_strength = (self.s_max - s + 1) / self.s_max
        new_pos = pos + swirl_strength * (nn_pos - pos) * np.random.uniform(-1, 1, self.dim)
        return np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])

    def crossover_middle(self, p):
        """ Crossover for middle population (blend of 3 individuals) """
        weights = np.random.dirichlet(np.ones(3))
        new_pos = weights[0]*p[0] + weights[1]*p[1] + weights[2]*p[2]
        return np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """ Run the Keshtel Algorithm """
        self.initialize_population()

        m1 = round(self.p1 * self.population_size)
        m2 = 2 * round((self.p2 * self.population_size) / 2)
        m3 = self.population_size - (m1 + m2)

        for iteration in range(self.max_iter):
            # N1 Best group
            lucky_keshtels = self.population[:m1]
            for k in range(m1):
                target = lucky_keshtels[k]
                rest = self.population[k+1:]
                target['nn'] = self.nearest_neighbor(target['position'], rest)

                S = 1
                while S <= 2 * self.s_max - 1:
                    candidate_pos = self.swirl(target, target['nn'], S)
                    candidate_cost = self.objective_function(candidate_pos)
                    if candidate_cost < target['cost']:
                        target['position'] = candidate_pos
                        target['cost'] = candidate_cost
                        target['nn'] = self.nearest_neighbor(target['position'], self.population)
                        S = 1
                    else:
                        S += 1
                lucky_keshtels[k] = target

            # N2 middle group
            pop_m2 = []
            for j in range(m2):
                temp_idx = j + m1
                i = np.random.choice([x for x in range(self.population_size) if x != temp_idx], 2, replace=False)
                p = [self.population[temp_idx]['position'],
                     self.population[i[0]]['position'],
                     self.population[i[1]]['position']]
                pos = self.crossover_middle(p)
                cost = self.objective_function(pos)
                pop_m2.append({'position': pos, 'cost': cost, 'nn': None})

            # N3 worst group
            pop_m3 = [{
                'position': self.random_solution(),
                'cost': self.objective_function(self.random_solution()),
                'nn': None
            } for _ in range(m3)]

            # Combine and sort
            self.population = lucky_keshtels + pop_m2 + pop_m3
            self.population.sort(key=lambda x: x['cost'])

            # Update best solution
            if self.population[0]['cost'] < self.best_value:
                self.best_solution = self.population[0]['position']
                self.best_value = self.population[0]['cost']

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

