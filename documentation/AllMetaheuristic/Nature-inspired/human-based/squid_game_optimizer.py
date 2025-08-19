import numpy as np

class SquidGameOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 attack_rate=0.5, defense_strength=0.3, fight_intensity=0.2, win_threshold=0.6):
        """
        Initialize the Squid Game Optimizer (SGO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of players (solutions).
        - max_iter: Maximum number of iterations.
        - attack_rate: Controls offensive movement intensity.
        - defense_strength: Controls defensive resistance.
        - fight_intensity: Controls randomness in fight simulation.
        - win_threshold: Threshold for determining winning state.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.attack_rate = attack_rate
        self.defense_strength = defense_strength
        self.fight_intensity = fight_intensity
        self.win_threshold = win_threshold

        self.players = None  # Population of players (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_players(self):
        """ Generate initial player positions randomly """
        self.players = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                        (self.population_size, self.dim))

    def evaluate_players(self):
        """ Compute fitness values for the players """
        return np.array([self.objective_function(player) for player in self.players])

    def divide_teams(self):
        """ Divide players into offensive and defensive teams """
        offensive_size = int(self.population_size * self.attack_rate)
        indices = np.random.permutation(self.population_size)
        offensive_indices = indices[:offensive_size]
        defensive_indices = indices[offensive_size:]
        return offensive_indices, defensive_indices

    def simulate_fight(self, offensive_idx, defensive_idx):
        """ Simulate fight between offensive and defensive players """
        r1 = np.random.rand()
        offensive_player = self.players[offensive_idx]
        defensive_player = self.players[defensive_idx]

        # Calculate movement vector based on fight intensity
        movement = self.fight_intensity * (offensive_player - defensive_player) * r1

        # Update offensive player position
        new_offensive = offensive_player + movement
        new_offensive = np.clip(new_offensive, self.bounds[:, 0], self.bounds[:, 1])

        # Update defensive player position with resistance
        resistance = self.defense_strength * (defensive_player - offensive_player) * (1 - r1)
        new_defensive = defensive_player + resistance
        new_defensive = np.clip(new_defensive, self.bounds[:, 0], self.bounds[:, 1])

        return new_offensive, new_defensive

    def determine_winners(self, offensive_indices, defensive_indices):
        """ Determine winners based on objective function values """
        fitness = self.evaluate_players()
        winners = []
        for off_idx, def_idx in zip(offensive_indices, defensive_indices):
            off_fitness = fitness[off_idx]
            def_fitness = fitness[def_idx]
            # Winner is determined if fitness improvement exceeds threshold
            if off_fitness < def_fitness * self.win_threshold:
                winners.append(off_idx)
            elif def_fitness < off_fitness * self.win_threshold:
                winners.append(def_idx)
        return winners

    def update_positions(self, winners):
        """ Update positions based on winning states """
        fitness = self.evaluate_players()
        for idx in range(self.population_size):
            if idx in winners:
                # Winners move towards best solution
                if self.best_solution is not None:
                    r2 = np.random.rand()
                    self.players[idx] += self.attack_rate * r2 * (self.best_solution - self.players[idx])
            else:
                # Losers undergo random perturbation
                r3 = np.random.rand(self.dim)
                self.players[idx] += self.fight_intensity * r3 * (self.bounds[:, 1] - self.bounds[:, 0])
            
            # Clip to bounds
            self.players[idx] = np.clip(self.players[idx], self.bounds[:, 0], self.bounds[:, 1])

    def optimize(self):
        """ Run the Squid Game Optimization """
        self.initialize_players()
        for iteration in range(self.max_iter):
            fitness = self.evaluate_players()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.players[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Divide players into offensive and defensive teams
            offensive_indices, defensive_indices = self.divide_teams()

            # Simulate fights between offensive and defensive players
            for off_idx, def_idx in zip(offensive_indices, defensive_indices[:len(offensive_indices)]):
                new_offensive, new_defensive = self.simulate_fight(off_idx, def_idx)
                self.players[off_idx] = new_offensive
                self.players[def_idx] = new_defensive

            # Determine winners and update positions
            winners = self.determine_winners(offensive_indices, defensive_indices)
            self.update_positions(winners)

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
