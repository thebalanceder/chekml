import numpy as np
import time

class WaterCycleAlgorithm:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100,
                 use_lsar_asp=True, max_lsar_iter=2000):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.max_iter = max_iter
        self.use_lsar_asp = use_lsar_asp
        self.max_lsar_iter = max_lsar_iter
        self.streams = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_streams(self):
        self.streams = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                         (self.population_size, self.dim))

    def evaluate_streams(self):
        return np.array([self.objective_function(stream) for stream in self.streams])

    def iop(self, C, D):
        if C.size == 0:
            return 0
        count = self.equivalent_class(C) - self.equivalent_class(np.hstack([C, D]))
        return np.sum(count)

    def equivalent_class(self, C):
        # Safely handle multidimensional structured views for uniqueness
        C_contig = np.ascontiguousarray(C)
        dtype = [('f{}'.format(i), C_contig.dtype) for i in range(C_contig.shape[1])]
        structured = C_contig.view(dtype).reshape(-1)
        _, inv, counts = np.unique(structured, return_inverse=True, return_counts=True)
        return counts[inv]

    def positive_region(self, C, D):
        if C.size == 0:
            return np.array([])
        count = self.equivalent_class(C) - self.equivalent_class(np.hstack([C, D]))
        return np.where(count == 0)[0]

    def random_select_att(self, B):
        if len(B) == 0:
            raise ValueError("Error: Empty input set")
        return np.random.choice(B)

    def fast_red(self, C, D):
        att = C.shape[1]
        iop_C = self.iop(C, D)
        w = np.zeros(att)
        for i in range(att):
            w[i] = self.iop(C[:, i:i + 1], D)
        ind = np.argsort(w)
        red = []
        for i in range(att):
            red = np.union1d(red, [ind[i]])
            if self.iop(C[:, red.astype(int)], D) == iop_C:
                break
        return red.astype(int)

    def aps_mechanism(self, C, D, B):
        u, v = 0, 0
        att = C.shape[1]
        pos = self.positive_region(C, D)
        unpos = np.setdiff1d(pos, self.positive_region(C[:, B], D))
        unred = np.setdiff1d(np.arange(att), B)
        add = []
        for k in unred:
            if len(self.positive_region(C[unpos][:, [k]], D[unpos])) == len(unpos):
                add = np.union1d(add, [k])
        for i in add:
            subspace = np.union1d(B, [i]).astype(int)
            unique_rows = np.unique(C[:, subspace], axis=0, return_index=True)[1]
            U = np.sort(unique_rows)
            for j in B:
                testB = np.setdiff1d(np.union1d(B, [i]), [j]).astype(int)
                if len(self.positive_region(C[U][:, testB], D[U])) == len(U):
                    v = i
                    u = j
                    break
            if u != 0 or v != 0:
                break
        return u, v

    def lsar(self, C, D, T):
        att = C.shape[1]
        red = self.fast_red(C, D)
        pos = self.positive_region(C, D)
        t = 0
        while t < T:
            a = self.random_select_att(red)
            if len(self.positive_region(C[:, np.setdiff1d(red, a)], D)) == len(pos):
                red = np.setdiff1d(red, a)
            else:
                u = self.random_select_att(np.setdiff1d(red, a))
                v = self.random_select_att(np.setdiff1d(np.arange(att), np.setdiff1d(red, a)))
                if len(self.positive_region(C[:, np.union1d(np.setdiff1d(np.setdiff1d(red, a), u), v)], D)) == len(pos):
                    red = np.union1d(np.setdiff1d(np.setdiff1d(red, a), u), v)
            t += 1
        return red, t

    def lsar_asp(self, C, D):
        RemoveSet = []
        red = self.fast_red(C, D)
        pos = self.positive_region(C, D)
        t = 0
        while len(red) != len(RemoveSet):
            a = self.random_select_att(np.setdiff1d(red, RemoveSet))
            if len(self.positive_region(C[:, np.setdiff1d(red, a)], D)) == len(pos):
                red = np.setdiff1d(red, a)
            else:
                u, v = self.aps_mechanism(C, D, np.setdiff1d(red, a))
                if u == 0 and v == 0:
                    RemoveSet = np.union1d(RemoveSet, a)
                else:
                    red = np.union1d(np.setdiff1d(np.setdiff1d(red, a), u), v)
                    RemoveSet = []
            t += 1
        return red, t

    def optimize(self):
        self.initialize_streams()
        for generation in range(self.max_iter):
            fitness = self.evaluate_streams()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.streams[min_idx].copy()
                self.best_value = fitness[min_idx]

            C = self.streams
            D = fitness.reshape(-1, 1)
            if self.use_lsar_asp:
                red, _ = self.lsar_asp(C, D)
            else:
                red, _ = self.lsar(C, D, self.max_lsar_iter)

            for i in range(self.population_size):
                if np.random.rand() < 0.5:
                    self.streams[i, red] = self.best_solution[red] + \
                        np.random.uniform(-1, 1, len(red)) * (self.bounds[red, 1] - self.bounds[red, 0]) * 0.1
                self.streams[i] = np.clip(self.streams[i], self.bounds[:, 0], self.bounds[:, 1])

            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

