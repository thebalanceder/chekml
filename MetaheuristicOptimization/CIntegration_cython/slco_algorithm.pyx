import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport ceil

# Ensure NumPy C API is initialized
cnp.import_array()

# Define types for performance
ctypedef cnp.double_t DTYPE_t
ctypedef cnp.npy_int INT_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class SoccerLeagueCompetitionOptimizer:
    # C-level attributes
    cdef public int dim, n_team, n_main_player, n_reserve_player, max_iter
    cdef public double alpha, beta, mutation_prob, s_say
    cdef public cnp.ndarray bounds
    cdef public object objective_function
    cdef public list league
    cdef public int best_team_idx
    cdef public double best_total_cost
    cdef public long n_eval

    def __init__(self, objective_function, int dim, bounds, int n_team=10, 
                 int n_main_player=5, int n_reserve_player=3, int max_iter=100, 
                 double alpha=2.0, double beta=2.0, double mutation_prob=0.02, 
                 double s_say=0.5):
        """
        Initialize the Soccer League Competition Optimizer (SLCO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - n_team: Number of teams in the league.
        - n_main_player: Number of main players per team.
        - n_reserve_player: Number of reserve players per team.
        - max_iter: Maximum number of iterations.
        - alpha: Coefficient for winner imitation (main players).
        - beta: Coefficient for winner imitation (main players).
        - mutation_prob: Probability of mutation in loser function.
        - s_say: Sorting sensitivity parameter.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)  # Bounds as [(low, high), ...]
        self.n_team = n_team
        self.n_main_player = n_main_player
        self.n_reserve_player = n_reserve_player
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.mutation_prob = mutation_prob
        self.s_say = s_say
        self.n_eval = 0  # Evaluation counter
        self.league = []
        self.best_team_idx = 0
        self.best_total_cost = np.inf

    cpdef void initialize_league(self):
        """Create initial league with random player positions."""
        cdef DTYPE_t[:, :] bounds = self.bounds
        cdef DTYPE_t[:] var_min = bounds[:, 0]
        cdef DTYPE_t[:] var_max = bounds[:, 1]
        cdef int i
        cdef dict team
        cdef list m_players, r_players
        cdef dict player

        for _ in range(self.n_team):
            m_players = []
            r_players = []
            for i in range(self.n_main_player):
                player = {
                    'Position': np.random.uniform(var_min, var_max, self.dim),
                    'Cost': 0.0
                }
                player['Cost'] = self.objective_function(player['Position'])
                self.n_eval += 1
                m_players.append(player)
            for i in range(self.n_reserve_player):
                player = {
                    'Position': np.random.uniform(var_min, var_max, self.dim),
                    'Cost': 0.0
                }
                player['Cost'] = self.objective_function(player['Position'])
                self.n_eval += 1
                r_players.append(player)
            team = {
                'MPlayer': m_players,
                'RPlayer': r_players,
                'TotalCost': 0.0
            }
            self.league.append(team)
        self.league = self.takhsis()
        self.league = self.update_total_cost()

    cpdef list takhsis(self):
        """Assign players to teams based on sorted costs."""
        cdef list all_players = []
        cdef dict team
        cdef list new_league = []
        cdef int kk = 0
        cdef cnp.ndarray player_costs
        cdef cnp.ndarray sort_order

        for team in self.league:
            all_players.extend(team['MPlayer'])
            all_players.extend(team['RPlayer'])
        
        player_costs = np.array([p['Cost'] for p in all_players], dtype=np.double)
        sort_order = np.argsort(player_costs)
        all_players = [all_players[i] for i in sort_order]

        for _ in range(self.n_team):
            team = {'MPlayer': [], 'RPlayer': [], 'TotalCost': 0.0}
            team['MPlayer'] = all_players[kk:kk + self.n_main_player]
            kk += self.n_main_player
            team['RPlayer'] = all_players[kk:kk + self.n_reserve_player]
            kk += self.n_reserve_player
            new_league.append(team)
        return new_league

    cpdef list update_total_cost(self):
        """Update total cost for each team based on main players' costs."""
        cdef int k
        cdef dict team
        cdef list costs
        cdef double mean_cost

        for k in range(self.n_team):
            team = self.league[k]
            costs = [player['Cost'] for player in team['MPlayer']]
            mean_cost = np.mean(costs)
            team['TotalCost'] = 1.0 / mean_cost if mean_cost != 0 else np.inf
            if team['TotalCost'] < self.best_total_cost:
                self.best_total_cost = team['TotalCost']
                self.best_team_idx = k
        return self.league

    cpdef tuple probability_host(self, int ii, int jj):
        """Determine winner and loser based on total cost probability."""
        cdef double total_cost_ii = self.league[ii]['TotalCost']
        cdef double total_cost_jj = self.league[jj]['TotalCost']
        cdef double p_simple = total_cost_ii / (total_cost_ii + total_cost_jj)
        cdef double r = <double>rand() / RAND_MAX
        if r < p_simple:
            return ii, jj
        return jj, ii

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void winner_function_main(self, int winner):
        """Update main players of the winner team."""
        cdef DTYPE_t[:, :] bounds = self.bounds
        cdef DTYPE_t[:] var_min = bounds[:, 0]
        cdef DTYPE_t[:] var_max = bounds[:, 1]
        cdef int i, j
        cdef DTYPE_t[:] w = np.random.uniform(0.7, 1.0, self.dim)
        cdef DTYPE_t[:] xxx_pos = np.zeros(self.dim, dtype=np.double)
        cdef double cost
        cdef dict xxx
        cdef list players
        cdef cnp.ndarray player_costs, sort_order

        for i in range(self.n_main_player):
            xxx = {'Position': xxx_pos, 'Cost': 0.0}
            for j in range(self.dim):
                xxx_pos[j] = (w[j] * self.league[winner]['MPlayer'][i]['Position'][j] +
                              self.alpha * (<double>rand() / RAND_MAX) * 
                              (self.league[winner]['MPlayer'][0]['Position'][j] - 
                               self.league[winner]['MPlayer'][i]['Position'][j]) +
                              self.beta * (<double>rand() / RAND_MAX) * 
                              (self.league[0]['MPlayer'][0]['Position'][j] - 
                               self.league[winner]['MPlayer'][i]['Position'][j]))
                xxx_pos[j] = min(max(xxx_pos[j], var_min[j]), var_max[j])
            cost = self.objective_function(xxx_pos)
            self.n_eval += 1

            if cost < self.league[winner]['MPlayer'][i]['Cost']:
                self.league[winner]['MPlayer'][i] = {'Position': np.asarray(xxx_pos).copy(), 'Cost': cost}
            else:
                w = np.random.uniform(0.0, 0.7, self.dim)
                for j in range(self.dim):
                    xxx_pos[j] = (w[j] * self.league[winner]['MPlayer'][i]['Position'][j] +
                                  self.alpha * (<double>rand() / RAND_MAX) * 
                                  (self.league[winner]['MPlayer'][0]['Position'][j] - 
                                   self.league[winner]['MPlayer'][i]['Position'][j]) +
                                  self.beta * (<double>rand() / RAND_MAX) * 
                                  (self.league[0]['MPlayer'][0]['Position'][j] - 
                                   self.league[winner]['MPlayer'][i]['Position'][j]))
                    xxx_pos[j] = min(max(xxx_pos[j], var_min[j]), var_max[j])
                cost = self.objective_function(xxx_pos)
                self.n_eval += 1
                if cost < self.league[winner]['MPlayer'][i]['Cost']:
                    self.league[winner]['MPlayer'][i] = {'Position': np.asarray(xxx_pos).copy(), 'Cost': cost}
                    players = self.league[winner]['MPlayer']
                    player_costs = np.array([p['Cost'] for p in players], dtype=np.double)
                    sort_order = np.argsort(player_costs)
                    self.league[winner]['MPlayer'] = [players[i] for i in sort_order]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void winner_function_reserve(self, int winner):
        """Update reserve players of the winner team."""
        cdef DTYPE_t[:, :] bounds = self.bounds
        cdef DTYPE_t[:] var_min = bounds[:, 0]
        cdef DTYPE_t[:] var_max = bounds[:, 1]
        cdef int i, j, jj
        cdef DTYPE_t[:] gravity = np.zeros(self.dim, dtype=np.double)
        cdef DTYPE_t[:] x, rq, beta
        cdef double cost
        cdef dict new_sol
        cdef int last_reserve_idx = self.n_reserve_player - 1

        for i in range(self.n_reserve_player):
            for j in range(self.dim):
                gravity[j] = 0.0
                for jj in range(self.n_main_player):
                    gravity[j] += self.league[winner]['MPlayer'][jj]['Position'][j]
                gravity[j] /= self.n_main_player

            x = self.league[winner]['RPlayer'][last_reserve_idx]['Position']
            beta = np.random.uniform(0.9, 1.0, self.dim)
            rq = np.zeros(self.dim, dtype=np.double)
            for j in range(self.dim):
                rq[j] = gravity[j] + beta[j] * (gravity[j] - x[j])
                rq[j] = min(max(rq[j], var_min[j]), var_max[j])
            cost = self.objective_function(rq)
            self.n_eval += 1

            if cost < self.league[winner]['RPlayer'][last_reserve_idx]['Cost']:
                self.league[winner]['RPlayer'][last_reserve_idx] = {'Position': np.asarray(rq).copy(), 'Cost': cost}
            else:
                beta = np.random.uniform(0.45, 0.45, self.dim)
                for j in range(self.dim):
                    rq[j] = gravity[j] + beta[j] * (x[j] - gravity[j])
                    rq[j] = min(max(rq[j], var_min[j]), var_max[j])
                cost = self.objective_function(rq)
                self.n_eval += 1
                if cost < self.league[winner]['RPlayer'][last_reserve_idx]['Cost']:
                    self.league[winner]['RPlayer'][last_reserve_idx] = {'Position': np.asarray(rq).copy(), 'Cost': cost}
                else:
                    new_sol = {'Position': np.random.uniform(var_min, var_max, self.dim),
                               'Cost': 0.0}
                    new_sol['Cost'] = self.objective_function(new_sol['Position'])
                    self.n_eval += 1
                    self.league[winner]['RPlayer'][last_reserve_idx] = new_sol

            all_players = self.league[winner]['MPlayer'] + self.league[winner]['RPlayer']
            player_costs = np.array([p['Cost'] for p in all_players], dtype=np.double)
            sort_order = np.argsort(player_costs)
            all_players = [all_players[i] for i in sort_order]
            self.league[winner]['MPlayer'] = all_players[:self.n_main_player]
            self.league[winner]['RPlayer'] = all_players[self.n_main_player:self.n_main_player + self.n_reserve_player]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void loser_function(self, int loser):
        """Update main and reserve players of the loser team."""
        cdef DTYPE_t[:, :] bounds = self.bounds
        cdef DTYPE_t[:] var_min = bounds[:, 0]
        cdef DTYPE_t[:] var_max = bounds[:, 1]
        cdef cnp.ndarray sigma = 0.1 * (np.asarray(var_max) - np.asarray(var_min))
        cdef int i, j, idx
        cdef cnp.ndarray[INT_t, ndim=1] indices = np.random.permutation(self.n_main_player).astype(np.int32)[:3]
        cdef dict player
        cdef DTYPE_t[:] new_pos, noise
        cdef double new_cost
        cdef list new_sols = []
        cdef cnp.ndarray[INT_t, ndim=1] r_indices
        cdef cnp.ndarray x1, x2, y1, y2, alpha
        cdef cnp.ndarray player_costs, sort_order
        cdef int n_mu

        # Mutate three random main players
        for idx in indices:
            player = self.league[loser]['MPlayer'][idx]
            n_mu = <int>ceil(self.mutation_prob * self.dim)
            r_indices = np.random.choice(self.dim, n_mu, replace=False).astype(np.int32)
            new_pos = np.asarray(player['Position']).copy()
            noise = sigma * np.random.randn(self.dim)
            for j in r_indices:
                new_pos[j] += noise[j]
            for j in range(self.dim):
                new_pos[j] = min(max(new_pos[j], var_min[j]), var_max[j])
            new_cost = self.objective_function(new_pos)
            self.n_eval += 1
            if new_cost < player['Cost']:
                self.league[loser]['MPlayer'][idx] = {'Position': new_pos.copy(), 'Cost': new_cost}

        # Crossover for reserve players
        for _ in range(self.n_reserve_player):
            r_indices = np.random.permutation(self.n_reserve_player).astype(np.int32)[:2]
            x1 = np.asarray(self.league[loser]['RPlayer'][r_indices[0]]['Position'])
            x2 = np.asarray(self.league[loser]['RPlayer'][r_indices[1]]['Position'])
            alpha = np.random.rand(self.dim)
            y1 = alpha * x1 + (1 - alpha) * x2
            y2 = alpha * x2 + (1 - alpha) * x1
            for j in range(self.dim):
                y1[j] = min(max(y1[j], var_min[j]), var_max[j])
                y2[j] = min(max(y2[j], var_min[j]), var_max[j])
            new_sols.append({'Position': y1.copy(), 'Cost': self.objective_function(y1)})
            new_sols.append({'Position': y2.copy(), 'Cost': self.objective_function(y2)})
            self.n_eval += 2

        # Sort and reassign players
        all_players = self.league[loser]['MPlayer'] + self.league[loser]['RPlayer'] + new_sols
        player_costs = np.array([p['Cost'] for p in all_players], dtype=np.double)
        sort_order = np.argsort(player_costs)
        all_players = [all_players[i] for i in sort_order]
        self.league[loser]['MPlayer'] = all_players[:self.n_main_player]
        self.league[loser]['RPlayer'] = all_players[self.n_main_player:self.n_main_player + self.n_reserve_player]

    cpdef void imitation(self, int winner):
        """Perform imitation phase for the winner."""
        self.winner_function_main(winner)
        self.winner_function_reserve(winner)

    cpdef void competition(self, int iteration):
        """Simulate competition between teams."""
        cdef int ii, jj
        cdef int winner, loser
        for ii in range(self.n_team - 2):
            for jj in range(ii + 1, self.n_team):
                winner, loser = self.probability_host(ii, jj)
                self.imitation(winner)
                self.loser_function(loser)
                self.league = self.update_total_cost()

    cpdef tuple optimize(self):
        """Run the Soccer League Competition Optimization."""
        self.initialize_league()
        cdef list history = []
        cdef int it
        cdef dict best_team
        cdef DTYPE_t[:] best_position
        cdef double best_cost

        for it in range(self.max_iter):
            self.competition(it)
            best_team = self.league[self.best_team_idx]
            best_position = best_team['MPlayer'][0]['Position']
            best_cost = best_team['MPlayer'][0]['Cost']
            history.append((it, np.asarray(best_position).copy(), best_cost))
            print(f"Iteration {it + 1}: Best Cost = {best_cost}")
        return np.asarray(best_position), best_cost, history
