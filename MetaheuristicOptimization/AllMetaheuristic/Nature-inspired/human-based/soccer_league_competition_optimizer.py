import numpy as np

class SoccerLeagueCompetitionOptimizer:
    def __init__(self, objective_function, dim, bounds, n_team=10, n_main_player=5, n_reserve_player=3, 
                 max_iter=100, alpha=2.0, beta=2.0, mutation_prob=0.02, s_say=0.5):
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
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.n_team = n_team
        self.n_main_player = n_main_player
        self.n_reserve_player = n_reserve_player
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.mutation_prob = mutation_prob
        self.s_say = s_say
        self.n_eval = 0  # Evaluation counter

        # League structure: List of teams, each with main and reserve players
        self.league = []
        self.best_team_idx = None
        self.best_total_cost = float('inf')

    def initialize_league(self):
        """Create initial league with random player positions."""
        var_min, var_max = self.bounds[:, 0], self.bounds[:, 1]
        for _ in range(self.n_team):
            team = {
                'MPlayer': [{'Position': np.random.uniform(var_min, var_max, self.dim),
                             'Cost': 0.0} for _ in range(self.n_main_player)],
                'RPlayer': [{'Position': np.random.uniform(var_min, var_max, self.dim),
                             'Cost': 0.0} for _ in range(self.n_reserve_player)],
                'TotalCost': 0.0
            }
            # Evaluate initial costs
            for player in team['MPlayer']:
                player['Cost'] = self.objective_function(player['Position'])
                self.n_eval += 1
            for player in team['RPlayer']:
                player['Cost'] = self.objective_function(player['Position'])
                self.n_eval += 1
            self.league.append(team)
        self.league = self.takhsis()
        self.league = self.update_total_cost()

    def takhsis(self):
        """Assign players to teams based on sorted costs."""
        # Collect all players
        all_players = []
        for team in self.league:
            all_players.extend(team['MPlayer'])
            all_players.extend(team['RPlayer'])
        
        # Sort players by cost
        player_costs = np.array([p['Cost'] for p in all_players])
        sort_order = np.argsort(player_costs)
        all_players = [all_players[i] for i in sort_order]

        # Reassign players to teams
        new_league = []
        kk = 0
        for _ in range(self.n_team):
            team = {'MPlayer': [], 'RPlayer': [], 'TotalCost': 0.0}
            # Assign main players
            team['MPlayer'] = all_players[kk:kk+self.n_main_player]
            kk += self.n_main_player
            # Assign reserve players
            team['RPlayer'] = all_players[kk:kk+self.n_reserve_player]
            kk += self.n_reserve_player
            new_league.append(team)
        return new_league

    def update_total_cost(self):
        """Update total cost for each team based on main players' costs."""
        for k, team in enumerate(self.league):
            costs = [player['Cost'] for player in team['MPlayer']]
            team['TotalCost'] = 1 / np.mean(costs) if np.mean(costs) != 0 else float('inf')
            if team['TotalCost'] < self.best_total_cost:
                self.best_total_cost = team['TotalCost']
                self.best_team_idx = k
        return self.league

    def probability_host(self, ii, jj):
        """Determine winner and loser based on total cost probability."""
        total_cost_ii = self.league[ii]['TotalCost']
        total_cost_jj = self.league[jj]['TotalCost']
        p_simple = total_cost_ii / (total_cost_ii + total_cost_jj)
        if np.random.rand() < p_simple:
            return ii, jj
        return jj, ii

    def winner_function_main(self, winner):
        """Update main players of the winner team."""
        var_min, var_max = self.bounds[:, 0], self.bounds[:, 1]
        for i in range(self.n_main_player):
            w = np.random.uniform(0.7, 1, self.dim)
            xxx = {'Position': np.zeros(self.dim), 'Cost': 0.0}
            xxx['Position'] = (w * self.league[winner]['MPlayer'][i]['Position'] +
                              self.alpha * np.random.rand() * (self.league[winner]['MPlayer'][0]['Position'] - 
                                                               self.league[winner]['MPlayer'][i]['Position']) +
                              self.beta * np.random.rand() * (self.league[0]['MPlayer'][0]['Position'] - 
                                                              self.league[winner]['MPlayer'][i]['Position']))
            xxx['Position'] = np.clip(xxx['Position'], var_min, var_max)
            xxx['Cost'] = self.objective_function(xxx['Position'])
            self.n_eval += 1

            if xxx['Cost'] < self.league[winner]['MPlayer'][i]['Cost']:
                self.league[winner]['MPlayer'][i] = xxx
            else:
                w = np.random.uniform(0, 0.7, self.dim)
                xxx['Position'] = (w * self.league[winner]['MPlayer'][i]['Position'] +
                                  self.alpha * np.random.rand() * (self.league[winner]['MPlayer'][0]['Position'] - 
                                                                   self.league[winner]['MPlayer'][i]['Position']) +
                                  self.beta * np.random.rand() * (self.league[0]['MPlayer'][0]['Position'] - 
                                                                  self.league[winner]['MPlayer'][i]['Position']))
                xxx['Position'] = np.clip(xxx['Position'], var_min, var_max)
                xxx['Cost'] = self.objective_function(xxx['Position'])
                self.n_eval += 1
                if xxx['Cost'] < self.league[winner]['MPlayer'][i]['Cost']:
                    self.league[winner]['MPlayer'][i] = xxx
                    # Sort main players by cost
                    players = self.league[winner]['MPlayer']
                    player_costs = [p['Cost'] for p in players]
                    sort_order = np.argsort(player_costs)
                    self.league[winner]['MPlayer'] = [players[i] for i in sort_order]

    def winner_function_reserve(self, winner):
        """Update reserve players of the winner team."""
        var_min, var_max = self.bounds[:, 0], self.bounds[:, 1]
        for i in range(self.n_reserve_player):
            # Compute gravity (mean position of main players)
            gravity = np.mean([p['Position'] for p in self.league[winner]['MPlayer']], axis=0)
            x = self.league[winner]['RPlayer'][-1]['Position']
            beta = np.random.uniform(0.9, 1, self.dim)
            rq = np.clip(gravity + beta * (gravity - x), var_min, var_max)
            new_sol = {'Position': rq, 'Cost': self.objective_function(rq)}
            self.n_eval += 1

            if new_sol['Cost'] < self.league[winner]['RPlayer'][-1]['Cost']:
                self.league[winner]['RPlayer'][-1] = new_sol
            else:
                beta = np.random.uniform(0.45, 0.45, self.dim)
                rq = np.clip(gravity + beta * (x - gravity), var_min, var_max)
                new_sol = {'Position': rq, 'Cost': self.objective_function(rq)}
                self.n_eval += 1
                if new_sol['Cost'] < self.league[winner]['RPlayer'][-1]['Cost']:
                    self.league[winner]['RPlayer'][-1] = new_sol
                else:
                    new_sol = {'Position': np.random.uniform(var_min, var_max, self.dim),
                               'Cost': 0.0}
                    new_sol['Cost'] = self.objective_function(new_sol['Position'])
                    self.n_eval += 1
                    self.league[winner]['RPlayer'][-1] = new_sol

            # Sort all players and reassign
            all_players = self.league[winner]['MPlayer'] + self.league[winner]['RPlayer']
            player_costs = [p['Cost'] for p in all_players]
            sort_order = np.argsort(player_costs)
            all_players = [all_players[i] for i in sort_order]
            self.league[winner]['MPlayer'] = all_players[:self.n_main_player]
            self.league[winner]['RPlayer'] = all_players[self.n_main_player:self.n_main_player+self.n_reserve_player]

    def loser_function(self, loser):
        """Update main and reserve players of the loser team."""
        var_min, var_max = self.bounds[:, 0], self.bounds[:, 1]
        sigma = 0.1 * (var_max - var_min)

        # Mutate three random main players
        indices = np.random.permutation(self.n_main_player)[:3]
        for idx in indices:
            player = self.league[loser]['MPlayer'][idx]
            n_mu = int(np.ceil(self.mutation_prob * self.dim))
            j = np.random.choice(self.dim, n_mu, replace=False)
            new_pos = player['Position'].copy()
            # Generate noise for the entire position vector
            noise = sigma * np.random.randn(self.dim)
            # Apply noise only to selected indices
            new_pos[j] += noise[j]
            new_pos = np.clip(new_pos, var_min, var_max)
            new_cost = self.objective_function(new_pos)
            self.n_eval += 1
            if new_cost < player['Cost']:
                self.league[loser]['MPlayer'][idx] = {'Position': new_pos, 'Cost': new_cost}

        # Crossover for reserve players
        new_sols = []
        for _ in range(self.n_reserve_player):
            indices = np.random.permutation(self.n_reserve_player)[:2]
            x1 = self.league[loser]['RPlayer'][indices[0]]['Position']
            x2 = self.league[loser]['RPlayer'][indices[1]]['Position']
            alpha = np.random.rand(self.dim)
            y1 = alpha * x1 + (1 - alpha) * x2
            y2 = alpha * x2 + (1 - alpha) * x1
            new_sols.append({'Position': np.clip(y1, var_min, var_max), 'Cost': self.objective_function(y1)})
            new_sols.append({'Position': np.clip(y2, var_min, var_max), 'Cost': self.objective_function(y2)})
            self.n_eval += 2

        # Sort and reassign players
        all_players = self.league[loser]['MPlayer'] + self.league[loser]['RPlayer'] + new_sols
        player_costs = [p['Cost'] for p in all_players]
        sort_order = np.argsort(player_costs)
        all_players = [all_players[i] for i in sort_order]
        self.league[loser]['MPlayer'] = all_players[:self.n_main_player]
        self.league[loser]['RPlayer'] = all_players[self.n_main_player:self.n_main_player+self.n_reserve_player]

    def imitation(self, winner):
        """Perform imitation phase for the winner."""
        self.winner_function_main(winner)
        self.winner_function_reserve(winner)

    def competition(self, iteration):
        """Simulate competition between teams."""
        for ii in range(self.n_team - 2):
            for jj in range(ii + 1, self.n_team):
                winner, loser = self.probability_host(ii, jj)
                self.imitation(winner)
                self.loser_function(loser)
                self.league = self.update_total_cost()

    def optimize(self):
        """Run the Soccer League Competition Optimization."""
        self.initialize_league()
        history = []
        for it in range(self.max_iter):
            self.competition(it)
            best_team = self.league[self.best_team_idx]
            best_position = best_team['MPlayer'][0]['Position']
            best_cost = best_team['MPlayer'][0]['Cost']
            history.append((it, best_position.copy(), best_cost))
            print(f"Iteration {it + 1}: Best Cost = {best_cost}")
        return best_position, best_cost, history
