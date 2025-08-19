import numpy as np

class CoyoteOptimization:
    def __init__(self, objective_function, dim, bounds, n_packs=20, n_coy=5, max_nfeval=20000):
        """
        Initialize the Coyote Optimization Algorithm (COA).

        Parameters:
        - objective_function: Function to optimize (minimize).
        - dim: Number of dimensions (variables).
        - bounds: Array of shape (2, dim) with [lower, upper] bounds for each dimension.
        - n_packs: Number of coyote packs.
        - n_coy: Number of coyotes per pack.
        - max_nfeval: Maximum number of function evaluations.
        """
        if n_coy < 3:
            raise Exception("At least 3 coyotes per pack must be used")

        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Shape: (2, dim)
        self.n_packs = n_packs
        self.n_coy = n_coy
        self.max_nfeval = max_nfeval
        self.p_leave = 0.01 * (n_coy ** 2)  # Increased for more pack exchanges
        self.Ps = 1 / dim  # Scattering probability
        self.pop_total = n_packs * n_coy

        self.coyotes = None  # Population of coyotes
        self.costs = None  # Costs of coyotes
        self.ages = None  # Ages of coyotes
        self.packs = None  # Pack assignments
        self.best_solution = None  # Global best solution
        self.best_value = float("inf")  # Global best cost
        self.nfeval = 0  # Function evaluation counter
        self.history = []  # Optimization history

    def initialize_coyotes(self):
        """ Initialize coyote population and packs """
        # Initialize coyotes within bounds (Eq. 2)
        self.coyotes = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_total, self.dim))
        self.costs = np.zeros((1, self.pop_total))
        self.ages = np.zeros((1, self.pop_total))
        self.packs = np.random.permutation(self.pop_total).reshape(self.n_packs, self.n_coy)

        # Evaluate initial coyotes (Eq. 3)
        for c in range(self.pop_total):
            self.costs[0, c] = self.objective_function(self.coyotes[c, :])
            self.nfeval += 1

        # Initialize best solution
        ibest = np.argmin(self.costs[0, :])
        self.best_value = self.costs[0, ibest]
        self.best_solution = self.coyotes[ibest, :].copy()

    def limit_bounds(self, X):
        """ Keep solutions within search space constraints """
        # Ensure X is a 2D array
        X = np.atleast_2d(X)  # Shape: (n, dim) or (1, dim) for single coyote
        X_clipped = np.clip(X, self.bounds[0], self.bounds[1])

        # Add small perturbation if at boundary to prevent sticking
        for i in range(X_clipped.shape[0]):
            for j in range(self.dim):
                if X_clipped[i, j] == self.bounds[0, j]:
                    X_clipped[i, j] += np.random.uniform(0, 0.1 * (self.bounds[1, j] - self.bounds[0, j]))
                elif X_clipped[i, j] == self.bounds[1, j]:
                    X_clipped[i, j] -= np.random.uniform(0, 0.1 * (self.bounds[1, j] - self.bounds[0, j]))

        # Return 1D array if input was 1D
        return X_clipped[0] if X.shape[0] == 1 else X_clipped

    def update_pack(self, pack_idx):
        """ Update coyotes within a pack """
        # Extract pack data
        pack_coyotes = self.coyotes[self.packs[pack_idx, :], :]
        pack_costs = self.costs[0, self.packs[pack_idx, :]]
        pack_ages = self.ages[0, self.packs[pack_idx, :]]

        # Identify alpha coyote (Eq. 5)
        ind = np.argsort(pack_costs)
        pack_costs = pack_costs[ind]
        pack_coyotes = pack_coyotes[ind, :]
        pack_ages = pack_ages[ind]
        c_alpha = pack_coyotes[0, :]

        # Compute social tendency (Eq. 6)
        tendency = np.median(pack_coyotes, 0)

        # Check population diversity
        diversity = np.std(pack_coyotes, axis=0).mean()
        scale_factor = 1.5 if diversity < 0.1 else 1.0  # Boost exploration if low diversity

        # Update coyotes' social conditions (Eq. 12)
        new_coyotes = np.zeros((self.n_coy, self.dim))
        for c in range(self.n_coy):
            rc1 = c
            while rc1 == c:
                rc1 = np.random.randint(self.n_coy)
            rc2 = c
            while rc2 == c or rc2 == rc1:
                rc2 = np.random.randint(self.n_coy)

            new_coyotes[c, :] = pack_coyotes[c, :] + \
                                scale_factor * np.random.rand() * (c_alpha - pack_coyotes[rc1, :]) + \
                                scale_factor * np.random.rand() * (tendency - pack_coyotes[rc2, :])

            # Ensure new solutions are within bounds
            new_coyotes[c, :] = self.limit_bounds(new_coyotes[c, :])

            # Evaluate new solution (Eq. 13)
            new_cost = self.objective_function(new_coyotes[c, :])
            self.nfeval += 1

            # Update if better (Eq. 14)
            if new_cost < pack_costs[c]:
                pack_costs[c] = new_cost
                pack_coyotes[c, :] = new_coyotes[c, :]

        # Birth of a new coyote (Eq. 7 and Alg. 1)
        parents = np.random.permutation(self.n_coy)[:2]
        prob1 = prob2 = (1 - self.Ps) / 2
        pdr = np.random.permutation(self.dim)
        p1 = np.zeros((1, self.dim))
        p2 = np.zeros((1, self.dim))
        p1[0, pdr[0]] = 1
        p2[0, pdr[1]] = 1
        r = np.random.rand(1, self.dim - 2)
        p1[0, pdr[2:]] = r < prob1
        p2[0, pdr[2:]] = r > 1 - prob2

        # Generate pup with noise
        n = np.logical_not(np.logical_or(p1, p2))
        pup = p1 * pack_coyotes[parents[0], :] + \
              p2 * pack_coyotes[parents[1], :] + \
              n * np.random.uniform(self.bounds[0], self.bounds[1], (1, self.dim))

        # Evaluate pup
        pup_cost = self.objective_function(pup[0, :])
        self.nfeval += 1

        # Replace worst coyote if pup is better
        worst = np.flatnonzero(pack_costs > pup_cost)
        if len(worst) > 0:
            older = np.argsort(pack_ages[worst])
            which = worst[older[::-1]]
            pack_coyotes[which[0], :] = pup[0, :]
            pack_costs[which[0]] = pup_cost
            pack_ages[which[0]] = 0

        # Update pack information
        self.coyotes[self.packs[pack_idx], :] = pack_coyotes
        self.costs[0, self.packs[pack_idx]] = pack_costs
        self.ages[0, self.packs[pack_idx]] = pack_ages

    def pack_exchange(self):
        """ Allow coyotes to leave and join packs (Eq. 4) """
        if self.n_packs > 1 and np.random.rand() < self.p_leave:
            rp = np.random.permutation(self.n_packs)[:2]
            rc = [np.random.randint(0, self.n_coy), np.random.randint(0, self.n_coy)]
            self.packs[rp[0], rc[0]], self.packs[rp[1], rc[1]] = \
                self.packs[rp[1], rc[1]], self.packs[rp[0], rc[0]]

    def optimize(self):
        """ Run the Coyote Optimization Algorithm """
        self.initialize_coyotes()
        year = 1

        while self.nfeval < self.max_nfeval:
            year += 1

            # Update each pack
            for p in range(self.n_packs):
                self.update_pack(p)

            # Perform pack exchange
            self.pack_exchange()

            # Update ages
            self.ages += 1

            # Update global best
            ibest = np.argmin(self.costs[0, :])
            if self.costs[0, ibest] < self.best_value:
                self.best_value = self.costs[0, ibest]
                self.best_solution = self.coyotes[ibest, :].copy()

            self.history.append((year, self.best_solution.copy(), self.best_value))
            print(f"Year {year}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

def Sphere(X):
    """ Sphere function for testing """
    return np.sum(X ** 2)

if __name__ == "__main__":
    import time

    # Objective function and problem setup
    fobj = Sphere
    dim = 10
    bounds = np.array([[-10] * dim, [10] * dim])

    # COA parameters
    n_packs = 20
    n_coy = 5
    max_nfeval = 20000
    n_exper = 3

    # Run experiments
    t = time.time()
    results = np.zeros(n_exper)
    for i in range(n_exper):
        optimizer = CoyoteOptimization(fobj, dim, bounds, n_packs, n_coy, max_nfeval)
        best_solution, best_value, history = optimizer.optimize()
        results[i] = best_value
        print(f"Experiment {i + 1}, Best: {best_value}, time (s): {time.time() - t}")
        t = time.time()

    # Display statistics
    print("Statistics (min., avg., median, max., std.)")
    print([np.min(results), np.mean(results), np.median(results), np.max(results), np.std(results)])
