import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

# Define NumPy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

class ImperialistCompetitiveAlgorithm:
    def __init__(self, objective_function, int dim, bounds, int num_countries=200, 
                 int num_initial_imperialists=8, int max_decades=2000, double revolution_rate=0.3, 
                 double assimilation_coeff=2.0, double assimilation_angle_coeff=0.5, 
                 double zeta=0.02, double damp_ratio=0.99, double uniting_threshold=0.02, 
                 bint stop_if_single_empire=False):
        """
        Initialize the Imperialist Competitive Algorithm (ICA) optimizer.

        Parameters:
        - objective_function: Function to optimize (returns cost; lower is better).
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - num_countries: Total number of initial countries.
        - num_initial_imperialists: Number of initial imperialists.
        - max_decades: Maximum number of iterations (decades).
        - revolution_rate: Rate of sudden socio-political changes in colonies.
        - assimilation_coeff: Coefficient for colony movement toward imperialist.
        - assimilation_angle_coeff: Coefficient for angular assimilation (not used in this version).
        - zeta: Weight of colonies' mean cost in total empire cost.
        - damp_ratio: Damping factor for revolution rate.
        - uniting_threshold: Threshold for uniting similar empires (as % of search space).
        - stop_if_single_empire: If True, stop when only one empire remains.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.num_countries = num_countries
        self.num_imperialists = num_initial_imperialists
        self.num_colonies = num_countries - num_initial_imperialists
        self.max_decades = max_decades
        self.revolution_rate = revolution_rate
        self.assimilation_coeff = assimilation_coeff
        self.assimilation_angle_coeff = assimilation_angle_coeff
        self.zeta = zeta
        self.damp_ratio = damp_ratio
        self.uniting_threshold = uniting_threshold
        self.stop_if_single_empire = stop_if_single_empire

        self.countries = None
        self.costs = None
        self.empires = []
        self.best_solution = None
        self.best_cost = float("inf")
        self.history = []

    @cython.boundscheck(True)
    @cython.wraparound(True)
    def generate_new_country(self, int num_countries):
        """Generate random countries within bounds."""
        cdef np.ndarray[DTYPE_t, ndim=1] var_min = self.bounds[:, 0]
        cdef np.ndarray[DTYPE_t, ndim=1] var_max = self.bounds[:, 1]
        return np.random.uniform(var_min, var_max, (num_countries, self.dim))

    @cython.boundscheck(True)
    @cython.wraparound(True)
    def create_initial_empires(self):
        """Initialize empires by assigning imperialists and colonies."""
        print("Creating initial empires...")
        cdef np.ndarray[long] sort_indices = np.argsort(self.costs)
        self.costs = self.costs[sort_indices]
        self.countries = self.countries[sort_indices]

        cdef np.ndarray[DTYPE_t, ndim=2] imperialists_pos = self.countries[:self.num_imperialists]
        cdef np.ndarray[DTYPE_t, ndim=1] imperialists_cost = self.costs[:self.num_imperialists]
        cdef np.ndarray[DTYPE_t, ndim=2] colonies_pos = self.countries[self.num_imperialists:]
        cdef np.ndarray[DTYPE_t, ndim=1] colonies_cost = self.costs[self.num_imperialists:]

        cdef double max_cost = np.max(imperialists_cost)
        cdef np.ndarray[DTYPE_t, ndim=1] power
        if max_cost > 0:
            power = 1.3 * max_cost - imperialists_cost
        else:
            power = 0.7 * max_cost - imperialists_cost

        cdef np.ndarray[long] num_colonies = np.round(power / np.sum(power) * self.num_colonies).astype(int)
        num_colonies[-1] = self.num_colonies - np.sum(num_colonies[:-1])

        cdef np.ndarray[long] random_indices = np.random.permutation(self.num_colonies)
        cdef int start_idx = 0
        self.empires = []
        cdef int i, n_cols
        for i in range(self.num_imperialists):
            n_cols = num_colonies[i]
            indices = random_indices[start_idx:start_idx + n_cols]
            start_idx += n_cols

            empire = {
                'ImperialistPosition': imperialists_pos[i].copy(),
                'ImperialistCost': imperialists_cost[i],
                'ColoniesPosition': colonies_pos[indices].copy() if n_cols > 0 else np.array([]).reshape(0, self.dim),
                'ColoniesCost': colonies_cost[indices].copy() if n_cols > 0 else np.array([]),
                'TotalCost': None
            }
            if n_cols == 0:
                empire['ColoniesPosition'] = self.generate_new_country(1)
                empire['ColoniesCost'] = np.array([self.objective_function(empire['ColoniesPosition'][0])])
            empire['TotalCost'] = empire['ImperialistCost'] + self.zeta * np.mean(empire['ColoniesCost']) if empire['ColoniesCost'].size > 0 else empire['ImperialistCost']
            self.empires.append(empire)
        print(f"Created {len(self.empires)} empires")

    @cython.boundscheck(True)
    @cython.wraparound(True)
    def assimilate_colonies(self, dict empire):
        """Move colonies toward their imperialist (assimilation policy)."""
        if empire['ColoniesPosition'].size == 0:
            return empire

        cdef int num_cols = empire['ColoniesPosition'].shape[0]
        cdef np.ndarray[DTYPE_t, ndim=2] vector = np.tile(empire['ImperialistPosition'], (num_cols, 1)) - empire['ColoniesPosition']
        empire['ColoniesPosition'] = empire['ColoniesPosition'] + 2 * self.assimilation_coeff * np.random.rand(num_cols, self.dim) * vector

        empire['ColoniesPosition'] = np.clip(empire['ColoniesPosition'], self.bounds[:, 0], self.bounds[:, 1])
        return empire

    @cython.boundscheck(True)
    @cython.wraparound(True)
    def revolve_colonies(self, dict empire):
        """Introduce sudden changes in some colonies (revolution)."""
        if empire['ColoniesCost'].size == 0:
            return empire

        cdef int num_revolving = int(np.round(self.revolution_rate * empire['ColoniesCost'].size))
        if num_revolving == 0:
            return empire

        cdef np.ndarray[long] indices = np.random.choice(empire['ColoniesCost'].size, num_revolving, replace=False)
        empire['ColoniesPosition'][indices] = self.generate_new_country(num_revolving)
        return empire

    @cython.boundscheck(True)
    @cython.wraparound(True)
    def possess_empire(self, dict empire):
        """Allow a colony to become the imperialist if it has a lower cost."""
        if empire['ColoniesCost'].size == 0:
            return empire

        cdef double min_colony_cost = np.min(empire['ColoniesCost'])
        cdef int best_colony_idx = np.argmin(empire['ColoniesCost'])
        if min_colony_cost < empire['ImperialistCost']:
            old_imp_pos = empire['ImperialistPosition'].copy()
            old_imp_cost = empire['ImperialistCost']
            empire['ImperialistPosition'] = empire['ColoniesPosition'][best_colony_idx].copy()
            empire['ImperialistCost'] = empire['ColoniesCost'][best_colony_idx]
            empire['ColoniesPosition'][best_colony_idx] = old_imp_pos
            empire['ColoniesCost'][best_colony_idx] = old_imp_cost
        return empire

    @cython.boundscheck(True)
    @cython.wraparound(True)
    def unite_similar_empires(self):
        """Merge empires that are too close."""
        print("Uniting similar empires...")
        cdef double threshold = self.uniting_threshold * sqrt(np.sum((self.bounds[:, 1] - self.bounds[:, 0]) ** 2))
        cdef int i = 0
        cdef int j, better_idx, worse_idx
        cdef double distance
        while i < len(self.empires) - 1:
            j = i + 1
            while j < len(self.empires):
                distance = sqrt(np.sum((self.empires[i]['ImperialistPosition'] - self.empires[j]['ImperialistPosition']) ** 2))
                if distance <= threshold:
                    if self.empires[i]['ImperialistCost'] < self.empires[j]['ImperialistCost']:
                        better_idx, worse_idx = i, j
                    else:
                        better_idx, worse_idx = j, i

                    worse_pos = self.empires[worse_idx]['ColoniesPosition']
                    worse_cost = self.empires[worse_idx]['ColoniesCost']
                    if worse_pos.shape[0] == worse_cost.size:
                        if worse_pos.size > 0:
                            self.empires[better_idx]['ColoniesPosition'] = np.vstack((
                                self.empires[better_idx]['ColoniesPosition'],
                                self.empires[worse_idx]['ImperialistPosition'].reshape(1, -1),
                                worse_pos
                            ))
                            self.empires[better_idx]['ColoniesCost'] = np.concatenate((
                                self.empires[better_idx]['ColoniesCost'],
                                [self.empires[worse_idx]['ImperialistCost']],
                                worse_cost
                            ))
                        else:
                            self.empires[better_idx]['ColoniesPosition'] = np.vstack((
                                self.empires[better_idx]['ColoniesPosition'],
                                self.empires[worse_idx]['ImperialistPosition'].reshape(1, -1)
                            ))
                            self.empires[better_idx]['ColoniesCost'] = np.concatenate((
                                self.empires[better_idx]['ColoniesCost'],
                                [self.empires[worse_idx]['ImperialistCost']]
                            ))
                        self.empires[better_idx]['TotalCost'] = self.empires[better_idx]['ImperialistCost'] + \
                            self.zeta * np.mean(self.empires[better_idx]['ColoniesCost']) if self.empires[better_idx]['ColoniesCost'].size > 0 else \
                            self.empires[better_idx]['ImperialistCost']
                        self.empires.pop(worse_idx)
                    else:
                        print(f"Warning: Mismatched shapes in unite_similar_empires: pos={worse_pos.shape}, cost={worse_cost.shape}")
                        j += 1
                        continue
                j += 1
            i += 1
        print(f"Empires after uniting: {len(self.empires)}")

    @cython.boundscheck(True)
    @cython.wraparound(True)
    def imperialistic_competition(self):
        """Perform competition among empires, transferring colonies from weakest to others."""
        print("Starting imperialistic competition...")
        if np.random.rand() > 0.11 or len(self.empires) <= 1:
            print("Skipping competition")
            return

        cdef np.ndarray[DTYPE_t, ndim=1] total_costs = np.array([empire['TotalCost'] for empire in self.empires])
        cdef double max_cost = np.max(total_costs)
        cdef np.ndarray[DTYPE_t, ndim=1] powers = max_cost - total_costs
        if np.sum(powers) == 0:
            print("No power difference, skipping")
            return
        cdef np.ndarray[DTYPE_t, ndim=1] possession_prob = powers / np.sum(powers)
        cdef np.ndarray[DTYPE_t, ndim=1] diff = possession_prob - np.random.rand(len(possession_prob))
        cdef int selected_idx = np.argmax(diff)
        cdef int weakest_idx = np.argmax(total_costs)

        if self.empires[weakest_idx]['ColoniesCost'].size == 0:
            print("Weakest empire has no colonies, skipping")
            return

        cdef int colony_idx = np.random.randint(0, self.empires[weakest_idx]['ColoniesCost'].size)
        if self.empires[weakest_idx]['ColoniesPosition'].shape[0] != self.empires[weakest_idx]['ColoniesCost'].size:
            print(f"Error: Mismatched shapes in imperialistic_competition: pos={self.empires[weakest_idx]['ColoniesPosition'].shape}, cost={self.empires[weakest_idx]['ColoniesCost'].shape}")
            return

        self.empires[selected_idx]['ColoniesPosition'] = np.vstack((
            self.empires[selected_idx]['ColoniesPosition'],
            self.empires[weakest_idx]['ColoniesPosition'][colony_idx].reshape(1, -1)
        ))
        self.empires[selected_idx]['ColoniesCost'] = np.append(
            self.empires[selected_idx]['ColoniesCost'],
            self.empires[weakest_idx]['ColoniesCost'][colony_idx]
        )
        self.empires[weakest_idx]['ColoniesPosition'] = np.delete(
            self.empires[weakest_idx]['ColoniesPosition'], colony_idx, axis=0
        )
        self.empires[weakest_idx]['ColoniesCost'] = np.delete(
            self.empires[weakest_idx]['ColoniesCost'], colony_idx
        )
        if self.empires[weakest_idx]['ColoniesCost'].size <= 1:
            if self.empires[weakest_idx]['ColoniesCost'].size == 1:
                if self.empires[weakest_idx]['ColoniesPosition'].shape[0] == 1:
                    self.empires[selected_idx]['ColoniesPosition'] = np.vstack((
                        self.empires[selected_idx]['ColoniesPosition'],
                        self.empires[weakest_idx]['ColoniesPosition']
                    ))
                    self.empires[selected_idx]['ColoniesCost'] = np.append(
                        self.empires[selected_idx]['ColoniesCost'],
                        self.empires[weakest_idx]['ColoniesCost']
                    )
            self.empires[selected_idx]['ColoniesPosition'] = np.vstack((
                self.empires[selected_idx]['ColoniesPosition'],
                self.empires[weakest_idx]['ImperialistPosition'].reshape(1, -1)
            ))
            self.empires[selected_idx]['ColoniesCost'] = np.append(
                self.empires[selected_idx]['ColoniesCost'],
                self.empires[weakest_idx]['ImperialistCost']
            )
            self.empires.pop(weakest_idx)
        print("Completed imperialistic competition")

    def optimize(self):
        """Run the Imperialist Competitive Algorithm."""
        print("Starting optimization...")
        self.countries = self.generate_new_country(self.num_countries)
        self.costs = np.array([self.objective_function(country) for country in self.countries])
        self.create_initial_empires()

        cdef int decade, i
        for decade in range(self.max_decades):
            self.revolution_rate *= self.damp_ratio
            for i in range(len(self.empires)):
                self.empires[i] = self.assimilate_colonies(self.empires[i])
                self.empires[i] = self.revolve_colonies(self.empires[i])
                if self.empires[i]['ColoniesPosition'].size > 0:
                    if self.empires[i]['ColoniesPosition'].shape[0] != self.empires[i]['ColoniesCost'].size:
                        print(f"Error: Mismatched shapes in optimize: pos={self.empires[i]['ColoniesPosition'].shape}, cost={self.empires[i]['ColoniesCost'].shape}")
                        return None, None, []
                    self.empires[i]['ColoniesCost'] = np.array([
                        self.objective_function(pos) for pos in self.empires[i]['ColoniesPosition']
                    ])
                self.empires[i] = self.possess_empire(self.empires[i])
                self.empires[i]['TotalCost'] = self.empires[i]['ImperialistCost'] + \
                    self.zeta * np.mean(self.empires[i]['ColoniesCost']) if self.empires[i]['ColoniesCost'].size > 0 else \
                    self.empires[i]['ImperialistCost']

            self.unite_similar_empires()
            self.imperialistic_competition()

            imperialists_costs = np.array([empire['ImperialistCost'] for empire in self.empires])
            min_idx = np.argmin(imperialists_costs)
            if imperialists_costs[min_idx] < self.best_cost:
                self.best_solution = self.empires[min_idx]['ImperialistPosition'].copy()
                self.best_cost = imperialists_costs[min_idx]

            self.history.append((decade, self.best_solution.copy(), self.best_cost))
            print(f"Decade {decade + 1}: Best Cost = {self.best_cost}")

            if len(self.empires) == 1 and self.stop_if_single_empire:
                break

        print("Optimization completed")
        return self.best_solution, self.best_cost, self.history

# Example usage:
if __name__ == "__main__":
    def benchmark_function(x, fun_number=6):
        return np.sum(x ** 2)

    dim = 30
    bounds = [(-6, 6)] * dim
    ica = ImperialistCompetitiveAlgorithm(
        objective_function=benchmark_function,
        dim=dim,
        bounds=bounds,
        num_countries=200,
        num_initial_imperialists=8,
        max_decades=2000
    )
    best_solution, best_cost, history = ica.optimize()
    print(f"Best Solution: {best_solution}")
    print(f"Best Cost: {best_cost}")
