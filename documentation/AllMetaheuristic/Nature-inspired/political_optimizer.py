import numpy as np

class PoliticalOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=64, parties=8, max_iter=100, lambda_rate=1.0):
        """
        Initialize the Political Optimizer (PO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Total number of search agents (parties * areas).
        - parties: Number of political parties.
        - max_iter: Maximum number of iterations.
        - lambda_rate: Maximum limit of party switching rate.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.parties = parties
        self.areas = population_size // parties  # Number of constituencies
        self.max_iter = max_iter
        self.lambda_rate = lambda_rate

        self.positions = None
        self.fitness = None
        self.aux_positions = None
        self.aux_fitness = None
        self.prev_positions = None
        self.prev_fitness = None
        self.a_winners = None
        self.a_winner_indices = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_population(self):
        """ Generate initial population randomly """
        self.positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                          (self.population_size, self.dim))
        self.aux_positions = self.positions.copy()
        self.prev_positions = self.positions.copy()
        self.fitness = np.array([self.objective_function(pos) for pos in self.positions])
        self.aux_fitness = self.fitness.copy()
        self.prev_fitness = self.fitness.copy()

    def election(self):
        """ Perform election phase to select constituency winners """
        self.a_winners = np.zeros((self.areas, self.dim))
        self.a_winner_indices = np.zeros(self.areas, dtype=int)
        for a in range(self.areas):
            start_idx = a
            end_idx = self.population_size
            step = self.areas
            constituency_indices = np.arange(start_idx, end_idx, step)
            constituency_fitness = self.fitness[constituency_indices]
            min_idx = np.argmin(constituency_fitness)
            self.a_winner_indices[a] = constituency_indices[min_idx]
            self.a_winners[a, :] = self.positions[self.a_winner_indices[a], :]

    def government_formation(self):
        """ Update positions based on government formation """
        for p in range(self.parties):
            party_start = p * self.areas
            party_indices = np.arange(party_start, party_start + self.areas)
            party_fitness = self.fitness[party_indices]
            party_leader_idx = party_indices[np.argmin(party_fitness)]
            party_leader_pos = self.positions[party_leader_idx, :]
            for a in range(self.areas):
                member_idx = party_start + a
                if member_idx != party_leader_idx:
                    member_pos = self.positions[member_idx, :]
                    new_pos = member_pos + np.random.rand() * (party_leader_pos - member_pos)
                    new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])
                    new_fitness = self.objective_function(new_pos)
                    if new_fitness < self.fitness[member_idx]:
                        self.positions[member_idx, :] = new_pos
                        self.fitness[member_idx] = new_fitness

    def election_campaign(self):
        """ Perform election campaign phase """
        for p in range(self.parties):
            party_start = p * self.areas
            for a in range(self.areas):
                member_idx = party_start + a
                current_pos = self.positions[member_idx, :]
                prev_pos = self.prev_positions[member_idx, :]
                new_pos = current_pos + np.random.rand(self.dim) * (current_pos - prev_pos)
                new_pos = np.clip(new_pos, self.bounds[:, 0], self.bounds[:, 1])
                new_fitness = self.objective_function(new_pos)
                if new_fitness < self.fitness[member_idx]:
                    self.positions[member_idx, :] = new_pos
                    self.fitness[member_idx] = new_fitness

    def party_switching(self, t):
        """ Perform party switching phase """
        psr = (1 - t * (1 / self.max_iter)) * self.lambda_rate
        for p in range(self.parties):
            for a in range(self.areas):
                from_idx = p * self.areas + a
                if np.random.rand() < psr:
                    to_party = np.random.randint(self.parties)
                    while to_party == p:
                        to_party = np.random.randint(self.parties)
                    to_start = to_party * self.areas
                    to_indices = np.arange(to_start, to_start + self.areas)
                    to_least_fit_idx = to_indices[np.argmax(self.fitness[to_indices])]
                    # Swap positions
                    self.positions[[to_least_fit_idx, from_idx], :] = self.positions[[from_idx, to_least_fit_idx], :]
                    # Swap fitness
                    self.fitness[[to_least_fit_idx, from_idx]] = self.fitness[[from_idx, to_least_fit_idx]]

    def parliamentarism(self):
        """ Perform parliamentarism phase """
        for a in range(self.areas):
            new_winner = self.a_winners[a, :].copy()
            winner_idx = self.a_winner_indices[a]
            to_area = np.random.randint(self.areas)
            while to_area == a:
                to_area = np.random.randint(self.areas)
            to_winner = self.a_winners[to_area, :]
            distance = np.abs(to_winner - new_winner)
            new_winner = to_winner + (2 * np.random.rand(self.dim) - 1) * distance
            new_winner = np.clip(new_winner, self.bounds[:, 0], self.bounds[:, 1])
            new_fitness = self.objective_function(new_winner)
            if new_fitness < self.fitness[winner_idx]:
                self.positions[winner_idx, :] = new_winner
                self.fitness[winner_idx] = new_fitness
                self.a_winners[a, :] = new_winner

    def optimize(self):
        """ Run the Political Optimizer """
        self.initialize_population()
        self.election()
        self.aux_fitness = self.fitness.copy()
        self.prev_fitness = self.fitness.copy()
        self.government_formation()

        self.history = []
        for t in range(self.max_iter):
            self.prev_fitness = self.aux_fitness.copy()
            self.prev_positions = self.aux_positions.copy()
            self.aux_fitness = self.fitness.copy()
            self.aux_positions = self.positions.copy()

            self.election_campaign()
            self.party_switching(t)
            self.election()
            self.government_formation()
            self.parliamentarism()

            min_idx = np.argmin(self.fitness)
            if self.fitness[min_idx] < self.best_value:
                self.best_solution = self.positions[min_idx, :].copy()
                self.best_value = self.fitness[min_idx]

            self.history.append((t, self.best_solution.copy(), self.best_value))
            print(f"Iteration {t + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
