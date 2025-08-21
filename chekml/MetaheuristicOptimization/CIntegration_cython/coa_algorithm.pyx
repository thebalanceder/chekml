# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport fabs
from libc.stdlib cimport rand, srand
from cython cimport boundscheck, wraparound

cdef double uniform(double a, double b):
    return a + (b - a) * np.random.rand()

cdef double[:, :] to_2d(np.ndarray arr):
    cdef double[:, :] out = arr
    return out

cdef class CoyoteOptimization:
    cdef:
        object objective_function
        int dim, n_packs, n_coy, max_nfeval, pop_total, nfeval
        double p_leave, Ps, best_value
        np.ndarray coyotes, costs, ages, bounds, best_solution
        np.ndarray packs
        list history

    def __init__(self, objective_function, int dim, bounds, int n_packs=20, int n_coy=5, int max_nfeval=20000):
        if n_coy < 3:
            raise ValueError("At least 3 coyotes per pack must be used")

        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.n_packs = n_packs
        self.n_coy = n_coy
        self.max_nfeval = max_nfeval
        self.pop_total = n_packs * n_coy
        self.p_leave = 0.01 * (n_coy ** 2)
        self.Ps = 1.0 / dim
        self.nfeval = 0
        self.best_value = 1e100
        self.history = []

    cpdef void initialize_coyotes(self):
        self.coyotes = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_total, self.dim)).astype(np.float64)
        self.costs = np.zeros((1, self.pop_total), dtype=np.float64)
        self.ages = np.zeros((1, self.pop_total), dtype=np.float64)
        self.packs = np.random.permutation(self.pop_total).reshape(self.n_packs, self.n_coy)

        for c in range(self.pop_total):
            self.costs[0, c] = self.objective_function(self.coyotes[c, :])
            self.nfeval += 1

        ibest = np.argmin(self.costs[0])
        self.best_value = self.costs[0, ibest]
        self.best_solution = self.coyotes[ibest, :].copy()

    cpdef np.ndarray limit_bounds(self, np.ndarray X):
        cdef int i, j
        cdef double[:] bmin = self.bounds[0]
        cdef double[:] bmax = self.bounds[1]

        X = np.clip(X, bmin, bmax)

        for i in range(X.shape[0]):
            for j in range(self.dim):
                if X[i, j] == bmin[j]:
                    X[i, j] += np.random.uniform(0, 0.1 * (bmax[j] - bmin[j]))
                elif X[i, j] == bmax[j]:
                    X[i, j] -= np.random.uniform(0, 0.1 * (bmax[j] - bmin[j]))

        return X

    cpdef void update_pack(self, int pack_idx):
        cdef int c, rc1, rc2, i
        indices = self.packs[pack_idx, :]
        pack_coyotes = self.coyotes[indices, :].copy()
        pack_costs = self.costs[0, indices].copy()
        pack_ages = self.ages[0, indices].copy()

        ind = np.argsort(pack_costs)
        pack_coyotes = pack_coyotes[ind, :]
        pack_costs = pack_costs[ind]
        pack_ages = pack_ages[ind]
        c_alpha = pack_coyotes[0, :].copy()
        tendency = np.median(pack_coyotes, axis=0)

        diversity = np.std(pack_coyotes, axis=0).mean()
        scale_factor = 1.5 if diversity < 0.1 else 1.0

        new_coyotes = np.zeros((self.n_coy, self.dim), dtype=np.float64)
        for c in range(self.n_coy):
            rc1, rc2 = c, c
            while rc1 == c:
                rc1 = np.random.randint(self.n_coy)
            while rc2 == c or rc2 == rc1:
                rc2 = np.random.randint(self.n_coy)

            new_coyotes[c, :] = pack_coyotes[c, :] + \
                scale_factor * np.random.rand() * (c_alpha - pack_coyotes[rc1, :]) + \
                scale_factor * np.random.rand() * (tendency - pack_coyotes[rc2, :])

            new_coyotes[c, :] = self.limit_bounds(new_coyotes[c: c + 1, :])[0]

            new_cost = self.objective_function(new_coyotes[c, :])
            self.nfeval += 1

            if new_cost < pack_costs[c]:
                pack_costs[c] = new_cost
                pack_coyotes[c, :] = new_coyotes[c, :]

        parents = np.random.permutation(self.n_coy)[:2]
        prob1 = prob2 = (1 - self.Ps) / 2
        pdr = np.random.permutation(self.dim)
        p1 = np.zeros(self.dim)
        p2 = np.zeros(self.dim)
        p1[pdr[0]] = 1
        p2[pdr[1]] = 1
        r = np.random.rand(self.dim - 2)
        p1[pdr[2:]] = r < prob1
        p2[pdr[2:]] = r > 1 - prob2
        n = np.logical_not(np.logical_or(p1, p2))

        pup = p1 * pack_coyotes[parents[0], :] + \
              p2 * pack_coyotes[parents[1], :] + \
              n * np.random.uniform(self.bounds[0], self.bounds[1], (self.dim,))
        pup_cost = self.objective_function(pup)
        self.nfeval += 1

        worst = np.where(pack_costs > pup_cost)[0]
        if len(worst) > 0:
            older = np.argsort(pack_ages[worst])
            which = worst[older[::-1]]
            pack_coyotes[which[0], :] = pup
            pack_costs[which[0]] = pup_cost
            pack_ages[which[0]] = 0

        self.coyotes[indices, :] = pack_coyotes
        self.costs[0, indices] = pack_costs
        self.ages[0, indices] = pack_ages

    cpdef void pack_exchange(self):
        if self.n_packs > 1 and np.random.rand() < self.p_leave:
            rp = np.random.permutation(self.n_packs)[:2]
            rc = [np.random.randint(0, self.n_coy), np.random.randint(0, self.n_coy)]
            self.packs[rp[0], rc[0]], self.packs[rp[1], rc[1]] = \
                self.packs[rp[1], rc[1]], self.packs[rp[0], rc[0]]

    cpdef tuple optimize(self):
        self.initialize_coyotes()
        cdef int year = 1
        cdef int p, ibest

        while self.nfeval < self.max_nfeval:
            year += 1
            for p in range(self.n_packs):
                self.update_pack(p)

            self.pack_exchange()
            self.ages += 1

            ibest = np.argmin(self.costs[0])
            if self.costs[0, ibest] < self.best_value:
                self.best_value = self.costs[0, ibest]
                self.best_solution = self.coyotes[ibest, :].copy()

            self.history.append((year, self.best_solution.copy(), self.best_value))
            print(f"Year {year}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

