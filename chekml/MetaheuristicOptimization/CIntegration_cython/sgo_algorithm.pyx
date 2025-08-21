# cython: language_level=3
# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport rand, RAND_MAX

# Define types for NumPy arrays
DTYPE = np.double
ctypedef np.double_t DTYPE_t

# Custom clip function
cdef inline double clip(double value, double min_val, double max_val):
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class SquidGameOptimizer:
    cdef public:
        object objective_function
        int dim
        np.ndarray bounds
        int population_size
        int max_iter
        double attack_rate
        double defense_strength
        double fight_intensity
        double win_threshold
        np.ndarray players
        np.ndarray best_solution
        double best_value
        list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100,
                 double attack_rate=0.5, double defense_strength=0.3, double fight_intensity=0.2,
                 double win_threshold=0.6):
        """
        Initialize the Squid Game Optimizer (SGO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Array of (lower, upper) bounds for each dimension.
        - population_size: Number of players (solutions).
        - max_iter: Maximum number of iterations.
        - attack_rate: Controls offensive movement intensity.
        - defense_strength: Controls defensive resistance.
        - fight_intensity: Controls randomness in fight simulation.
        - win_threshold: Threshold for determining winning state.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.attack_rate = attack_rate
        self.defense_strength = defense_strength
        self.fight_intensity = fight_intensity
        self.win_threshold = win_threshold
        self.players = None
        self.best_solution = None
        self.best_value = np.inf
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void initialize_players(self):
        """ Generate initial player positions randomly """
        cdef np.ndarray[DTYPE_t, ndim=2] players
        cdef np.ndarray[DTYPE_t, ndim=1] lower_bounds = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] upper_bounds = self.bounds[:, 1]
        players = np.random.uniform(lower_bounds, upper_bounds, (self.population_size, self.dim))
        self.players = players

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[DTYPE_t, ndim=1] evaluate_players(self):
        """ Compute fitness values for the players """
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(self.population_size, dtype=DTYPE)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.players[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple divide_teams(self):
        """ Divide players into offensive and defensive teams """
        cdef int offensive_size = int(self.population_size * self.attack_rate)
        cdef np.ndarray[np.int32_t, ndim=1] indices = np.random.permutation(self.population_size).astype(np.int32)
        cdef np.ndarray[np.int32_t, ndim=1] offensive_indices = indices[:offensive_size]
        cdef np.ndarray[np.int32_t, ndim=1] defensive_indices = indices[offensive_size:]
        return offensive_indices, defensive_indices

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple simulate_fight(self, int offensive_idx, int defensive_idx):
        """ Simulate fight between offensive and defensive players """
        cdef double r1 = <double>rand() / RAND_MAX
        cdef np.ndarray[DTYPE_t, ndim=1] offensive_player = self.players[offensive_idx]
        cdef np.ndarray[DTYPE_t, ndim=1] defensive_player = self.players[defensive_idx]
        cdef np.ndarray[DTYPE_t, ndim=1] movement, resistance, new_offensive, new_defensive
        cdef int i

        # Calculate movement vector based on fight intensity
        movement = np.empty(self.dim, dtype=DTYPE)
        for i in range(self.dim):
            movement[i] = self.fight_intensity * (offensive_player[i] - defensive_player[i]) * r1

        # Update offensive player position
        new_offensive = np.empty(self.dim, dtype=DTYPE)
        for i in range(self.dim):
            new_offensive[i] = offensive_player[i] + movement[i]
            new_offensive[i] = clip(new_offensive[i], self.bounds[i, 0], self.bounds[i, 1])

        # Update defensive player position with resistance
        resistance = np.empty(self.dim, dtype=DTYPE)
        for i in range(self.dim):
            resistance[i] = self.defense_strength * (defensive_player[i] - offensive_player[i]) * (1 - r1)

        new_defensive = np.empty(self.dim, dtype=DTYPE)
        for i in range(self.dim):
            new_defensive[i] = defensive_player[i] + resistance[i]
            new_defensive[i] = clip(new_defensive[i], self.bounds[i, 0], self.bounds[i, 1])

        return new_offensive, new_defensive

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef list determine_winners(self, np.ndarray[np.int32_t, ndim=1] offensive_indices,
                                 np.ndarray[np.int32_t, ndim=1] defensive_indices):
        """ Determine winners based on objective function values """
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = self.evaluate_players()
        cdef list winners = []
        cdef int off_idx, def_idx
        cdef double off_fitness, def_fitness
        cdef int i

        for i in range(min(len(offensive_indices), len(defensive_indices))):
            off_idx = offensive_indices[i]
            def_idx = defensive_indices[i]
            off_fitness = fitness[off_idx]
            def_fitness = fitness[def_idx]
            # Winner is determined if fitness improvement exceeds threshold
            if off_fitness < def_fitness * self.win_threshold:
                winners.append(off_idx)
            elif def_fitness < off_fitness * self.win_threshold:
                winners.append(def_idx)
        return winners

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_positions(self, list winners):
        """ Update positions based on winning states """
        cdef np.ndarray[DTYPE_t, ndim=1] fitness = self.evaluate_players()
        cdef np.ndarray[DTYPE_t, ndim=1] player, r3
        cdef double r2
        cdef int idx, i
        cdef bint is_winner

        for idx in range(self.population_size):
            is_winner = idx in winners
            player = self.players[idx]
            if is_winner and self.best_solution is not None:
                # Winners move towards best solution
                r2 = <double>rand() / RAND_MAX
                for i in range(self.dim):
                    player[i] += self.attack_rate * r2 * (self.best_solution[i] - player[i])
            else:
                # Losers undergo random perturbation
                r3 = np.random.rand(self.dim).astype(DTYPE)
                for i in range(self.dim):
                    player[i] += self.fight_intensity * r3[i] * (self.bounds[i, 1] - self.bounds[i, 0])

            # Clip to bounds
            for i in range(self.dim):
                player[i] = clip(player[i], self.bounds[i, 0], self.bounds[i, 1])
            self.players[idx] = player

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self):
        """ Run the Squid Game Optimization """
        self.initialize_players()
        cdef np.ndarray[DTYPE_t, ndim=1] fitness
        cdef int min_idx, iteration, off_idx, def_idx
        cdef np.ndarray[np.int32_t, ndim=1] offensive_indices, defensive_indices
        cdef np.ndarray[DTYPE_t, ndim=1] new_offensive, new_defensive
        cdef list winners

        for iteration in range(self.max_iter):
            fitness = self.evaluate_players()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.players[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Divide players into offensive and defensive teams
            offensive_indices, defensive_indices = self.divide_teams()

            # Simulate fights between offensive and defensive players
            for i in range(min(len(offensive_indices), len(defensive_indices))):
                off_idx = offensive_indices[i]
                def_idx = defensive_indices[i]
                new_offensive, new_defensive = self.simulate_fight(off_idx, def_idx)
                self.players[off_idx] = new_offensive
                self.players[def_idx] = new_defensive

            # Determine winners and update positions
            winners = self.determine_winners(offensive_indices, defensive_indices)
            self.update_positions(winners)

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
