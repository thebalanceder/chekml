# cython: language_level=3
# distutils: language=c

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport abs, sqrt

# Ensure NumPy C API is available
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class FlyingFoxOptimizer:
    cdef object objective_function
    cdef int dim
    cdef cnp.ndarray bounds
    cdef int population_size
    cdef int max_evals
    cdef cnp.ndarray deltasO
    cdef cnp.ndarray deltasO_max
    cdef cnp.ndarray deltasO_min
    cdef cnp.ndarray alpha_params
    cdef cnp.ndarray pa_params
    cdef int surv_list_size
    cdef list flying_foxes
    cdef cnp.ndarray best_solution
    cdef double best_value
    cdef double worst_value
    cdef list survival_list
    cdef list history
    cdef int func_count

    def __init__(self, objective_function, int dim, bounds, population_size=None, int max_evals=100000,
                 cnp.ndarray[double, ndim=1] deltasO=None, cnp.ndarray[double, ndim=1] alpha_params=None,
                 cnp.ndarray[double, ndim=1] pa_params=None):
        """
        Initialize the Flying Fox Optimization (FFO) algorithm.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of flying foxes (solutions). If None, calculated based on dim.
        - max_evals: Maximum number of function evaluations.
        - deltasO: Delta parameters for exploration control.
        - alpha_params: Alpha parameters for movement scaling.
        - pa_params: Probability parameters for local search.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.population_size = population_size if population_size is not None else int(10 + 2 * sqrt(dim))
        self.max_evals = max_evals
        self.deltasO = deltasO if deltasO is not None else np.array([0.2, 0.4, 0.6], dtype=np.double)
        self.deltasO_max = self.deltasO.copy()
        self.deltasO_min = self.deltasO / 10
        self.alpha_params = alpha_params if alpha_params is not None else np.array([1, 1.5, 1.9], dtype=np.double)
        self.pa_params = pa_params if pa_params is not None else np.array([0.5, 0.85, 0.99], dtype=np.double)
        self.surv_list_size = self.population_size // 4

        self.flying_foxes = None
        self.best_solution = None
        self.best_value = float("inf")
        self.worst_value = float("-inf")
        self.survival_list = []
        self.history = []
        self.func_count = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def initialize_population(self):
        """ Generate initial population of flying foxes """
        cdef cnp.ndarray[double, ndim=2] positions = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dim))
        cdef int i
        cdef double cost
        self.flying_foxes = []
        for i in range(self.population_size):
            cost = self.objective_function(positions[i])
            self.func_count += 1
            self.flying_foxes.append({
                "position": positions[i].copy(),
                "cost": cost,
                "past_cost": cost
            })

        if not self.flying_foxes:
            raise ValueError("Failed to initialize population")
        self.survival_list = self.flying_foxes[:self.surv_list_size]
        costs = [fox["cost"] for fox in self.flying_foxes]
        self.worst_value = max(costs) if costs else float("inf")
        min_idx = np.argmin(costs) if costs else 0
        self.best_solution = self.flying_foxes[min_idx]["position"].copy() if self.flying_foxes else np.zeros(self.dim)
        self.best_value = self.flying_foxes[min_idx]["cost"] if self.flying_foxes else float("inf")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def evaluate_population(self):
        """ Compute fitness values for the population """
        cdef int i
        cdef double cost
        costs = np.empty(self.population_size, dtype=np.double)
        for i in range(self.population_size):
            cost = self.objective_function(self.flying_foxes[i]["position"])
            costs[i] = cost
            self.flying_foxes[i]["cost"] = cost
            self.flying_foxes[i]["past_cost"] = cost
        self.func_count += self.population_size
        return costs

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fuzzy_self_tuning(self, dict fox, cnp.ndarray[double, ndim=1] best_solution, double best_cost, double worst_cost):
        """ Fuzzy self-tuning for alpha and pa parameters """
        cdef double delta = abs(best_cost - fox["cost"])
        cdef double deltamax = abs(best_cost - worst_cost)
        cdef double fi = (fox["cost"] - fox["past_cost"]) / deltamax if deltamax != 0 else 0
        cdef cnp.ndarray[double, ndim=1] deltas = self.deltasO * deltamax

        # Delta membership
        membership = {
            "delta": {"Same": {"Logic": False, "Value": 0.0},
                      "Near": {"Logic": False, "Value": 0.0},
                      "Far": {"Logic": False, "Value": 0.0}},
            "fi": {"Better": {"Logic": False, "Value": 0.0},
                   "Same": {"Logic": True, "Value": 1.0 - abs(fi)},
                   "Worse": {"Logic": False, "Value": 0.0}}
        }

        if 0 <= delta < deltas[1]:
            membership["delta"]["Far"]["Logic"] = False
            membership["delta"]["Far"]["Value"] = 0.0
            if delta < deltas[0]:
                membership["delta"]["Same"]["Logic"] = True
                membership["delta"]["Same"]["Value"] = 1.0
            else:
                membership["delta"]["Same"]["Logic"] = True
                membership["delta"]["Same"]["Value"] = (deltas[1] - delta) / (deltas[1] - deltas[0])
                membership["delta"]["Near"]["Logic"] = True
                membership["delta"]["Near"]["Value"] = (delta - deltas[0]) / (deltas[1] - deltas[0])
        elif deltas[1] <= delta <= deltamax:
            if deltas[1] <= delta <= deltas[2]:
                membership["delta"]["Near"]["Logic"] = True
                membership["delta"]["Near"]["Value"] = (deltas[2] - delta) / (deltas[2] - deltas[1])
                membership["delta"]["Far"]["Logic"] = True
                membership["delta"]["Far"]["Value"] = (delta - deltas[1]) / (deltas[2] - deltas[1])
            else:
                membership["delta"]["Near"]["Logic"] = False
                membership["delta"]["Near"]["Value"] = 0.0
                membership["delta"]["Far"]["Logic"] = True
                membership["delta"]["Far"]["Value"] = 1.0

        if -1 <= fi <= 1:
            if -1 <= fi < 0:
                membership["fi"]["Better"]["Logic"] = True
                membership["fi"]["Better"]["Value"] = -fi if fi > -1 else 1.0
            elif 0 < fi <= 1:
                membership["fi"]["Worse"]["Logic"] = True
                membership["fi"]["Worse"]["Value"] = fi if fi < 1 else 1.0

        # Alpha rules
        cdef cnp.ndarray[double, ndim=1] ruleno_alpha = np.zeros(3, dtype=np.double)
        ruleno_alpha[0] = membership["fi"]["Better"]["Value"] if membership["fi"]["Better"]["Logic"] else 0.0
        ruleno_alpha[1] = max(membership["fi"]["Same"]["Value"] if membership["fi"]["Same"]["Logic"] else 0.0,
                              membership["delta"]["Same"]["Value"] if membership["delta"]["Same"]["Logic"] else 0.0,
                              membership["delta"]["Near"]["Value"] if membership["delta"]["Near"]["Logic"] else 0.0)
        ruleno_alpha[2] = max(membership["fi"]["Worse"]["Value"] if membership["fi"]["Worse"]["Logic"] else 0.0,
                              membership["delta"]["Far"]["Value"] if membership["delta"]["Far"]["Logic"] else 0.0)
        cdef double alpha = np.sum(ruleno_alpha * self.alpha_params) / np.sum(ruleno_alpha) if np.sum(ruleno_alpha) != 0 else 1.0

        # Pa rules
        cdef cnp.ndarray[double, ndim=1] ruleno_pa = np.zeros(3, dtype=np.double)
        ruleno_pa[0] = max(membership["fi"]["Worse"]["Value"] if membership["fi"]["Worse"]["Logic"] else 0.0,
                           membership["delta"]["Far"]["Value"] if membership["delta"]["Far"]["Logic"] else 0.0)
        ruleno_pa[1] = max(membership["fi"]["Same"]["Value"] if membership["fi"]["Same"]["Logic"] else 0.0,
                           membership["delta"]["Same"]["Value"] if membership["delta"]["Same"]["Logic"] else 0.0)
        ruleno_pa[2] = max(membership["fi"]["Better"]["Value"] if membership["fi"]["Better"]["Logic"] else 0.0,
                           membership["delta"]["Near"]["Value"] if membership["delta"]["Near"]["Logic"] else 0.0)
        cdef double pa = np.sum(ruleno_pa * self.pa_params) / np.sum(ruleno_pa) if np.sum(ruleno_pa) != 0 else 0.85

        return alpha, pa

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_survival_list(self, cnp.ndarray[double, ndim=1] new_solution, double new_cost):
        """ Update survival list with new solution """
        temp = {"position": new_solution.copy(), "cost": new_cost, "past_cost": new_cost}
        if len(self.survival_list) == 0 or (len(self.survival_list) > 0 and new_cost < self.survival_list[len(self.survival_list)-1]["cost"]):
            self.survival_list.append(temp)
            self.survival_list.sort(key=lambda x: x["cost"])
            self.survival_list = self.survival_list[:self.surv_list_size]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def replace_with_survival_list(self):
        """ Replace a solution with a combination from survival list """
        if len(self.survival_list) < 2:
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
        cdef int m = np.random.randint(2, len(self.survival_list) + 1)
        cdef cnp.ndarray[cnp.intp_t, ndim=1] indices = np.random.choice(len(self.survival_list), m, replace=False)
        cdef cnp.ndarray[double, ndim=1] new_position = np.mean(
            [self.survival_list[i]["position"] for i in indices], axis=0)
        return np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def crossover(self, cnp.ndarray[double, ndim=1] parent1, cnp.ndarray[double, ndim=1] parent2):
        """ Perform crossover between two parents """
        cdef double extracros = 0.0
        cdef cnp.ndarray[double, ndim=1] L = np.random.uniform(-extracros, 1 + extracros, self.dim)
        cdef cnp.ndarray[double, ndim=1] off1 = L * parent1 + (1 - L) * parent2
        cdef cnp.ndarray[double, ndim=1] off2 = L * parent2 + (1 - L) * parent1
        return np.clip(off1, self.bounds[:, 0], self.bounds[:, 1]), np.clip(off2, self.bounds[:, 0], self.bounds[:, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """ Run the Flying Fox Optimization algorithm """
        print("Starting optimization...")
        self.initialize_population()
        print("Population initialized")
        cdef int iteration = 0
        cdef int i, j, j0
        cdef double alpha, pa, new_cost
        cdef cnp.ndarray[double, ndim=1] z, stepsize
        cdef cnp.ndarray[cnp.intp_t, ndim=1] A
        cdef cnp.ndarray[double, ndim=1] off1, off2
        cdef double deltamax
        cdef cnp.ndarray[double, ndim=1] deltas
        cdef list best_indices
        cdef int n_best
        cdef double p_death
        cdef int parent1, parent2
        cdef dict fox

        while self.func_count < self.max_evals:
            iteration += 1
            print(f"Starting iteration {iteration}")
            for i in range(self.population_size):
                print(f"Processing fox {i}")
                fox = self.flying_foxes[i]
                deltamax = abs(self.best_value - self.worst_value)
                deltas = self.deltasO * deltamax
                alpha, pa = self.fuzzy_self_tuning(fox, self.best_solution, self.best_value, self.worst_value)

                if abs(fox["cost"] - self.best_value) > (deltas[0] * 0.5):
                    z = fox["position"] + alpha * np.random.rand(self.dim) * (self.best_solution - fox["position"])
                else:
                    A = np.random.permutation(self.population_size)
                    A = A[A != i][:2]
                    stepsize = np.random.rand(self.dim) * (self.best_solution - fox["position"]) + \
                               np.random.rand(self.dim) * (self.flying_foxes[A[0]]["position"] - self.flying_foxes[A[1]]["position"])
                    z = fox["position"].copy()
                    j0 = np.random.randint(self.dim)
                    for j in range(self.dim):
                        if j == j0 or np.random.rand() >= pa:
                            z[j] += stepsize[j]

                z = np.clip(z, self.bounds[:, 0], self.bounds[:, 1])
                new_cost = self.objective_function(z)
                self.func_count += 1

                if new_cost < fox["cost"]:
                    self.flying_foxes[i] = {"position": z.copy(), "cost": new_cost, "past_cost": fox["cost"]}
                    if new_cost < self.best_value:
                        self.best_solution = z.copy()
                        self.best_value = new_cost

                if new_cost > self.worst_value:
                    self.worst_value = new_cost

                self.update_survival_list(z, new_cost)

                if abs(new_cost - self.best_value) > deltas[2]:
                    self.flying_foxes[i] = {
                        "position": self.replace_with_survival_list(),
                        "cost": 0.0,
                        "past_cost": 0.0
                    }
                    self.flying_foxes[i]["cost"] = self.objective_function(self.flying_foxes[i]["position"])
                    self.flying_foxes[i]["past_cost"] = self.flying_foxes[i]["cost"]
                    self.func_count += 1
                    if self.flying_foxes[i]["cost"] < self.best_value:
                        self.best_solution = self.flying_foxes[i]["position"].copy()
                        self.best_value = self.flying_foxes[i]["cost"]
                    if self.flying_foxes[i]["cost"] > self.worst_value:
                        self.worst_value = self.flying_foxes[i]["cost"]

            # Suffocating Flying Foxes
            best_indices = [i for i, fox in enumerate(self.flying_foxes) if fox["cost"] == self.best_value]
            n_best = len(best_indices)
            p_death = (n_best - 1) / self.population_size if self.population_size > 0 else 0

            for i in range(0, n_best, 2):
                if np.random.rand() < p_death:
                    if i == n_best - 1 and n_best % 2 == 1:
                        self.flying_foxes[best_indices[i]] = {
                            "position": self.replace_with_survival_list(),
                            "cost": 0.0,
                            "past_cost": 0.0
                        }
                        self.flying_foxes[best_indices[i]]["cost"] = self.objective_function(
                            self.flying_foxes[best_indices[i]]["position"])
                        self.flying_foxes[best_indices[i]]["past_cost"] = self.flying_foxes[best_indices[i]]["cost"]
                        self.func_count += 1
                        self.update_survival_list(
                            self.flying_foxes[best_indices[i]]["position"],
                            self.flying_foxes[best_indices[i]]["cost"])
                    else:
                        parent1, parent2 = np.random.randint(self.population_size, size=2)
                        if np.random.rand() < 0.5 and self.flying_foxes[parent1]["cost"] != self.flying_foxes[parent2]["cost"]:
                            off1, off2 = self.crossover(
                                self.flying_foxes[parent1]["position"],
                                self.flying_foxes[parent2]["position"])
                        else:
                            off1 = self.replace_with_survival_list()
                            off2 = self.replace_with_survival_list()

                        self.flying_foxes[best_indices[i]] = {
                            "position": off1.copy(),
                            "cost": 0.0,
                            "past_cost": 0.0
                        }
                        self.flying_foxes[best_indices[i]]["cost"] = self.objective_function(off1)
                        self.flying_foxes[best_indices[i]]["past_cost"] = self.flying_foxes[best_indices[i]]["cost"]
                        self.func_count += 1
                        self.update_survival_list(off1, self.flying_foxes[best_indices[i]]["cost"])

                        if i + 1 < n_best:
                            self.flying_foxes[best_indices[i + 1]] = {
                                "position": off2.copy(),
                                "cost": 0.0,
                                "past_cost": 0.0
                            }
                            self.flying_foxes[best_indices[i + 1]]["cost"] = self.objective_function(off2)
                            self.flying_foxes[best_indices[i + 1]]["past_cost"] = self.flying_foxes[best_indices[i + 1]]["cost"]
                            self.func_count += 1
                            self.update_survival_list(off2, self.flying_foxes[best_indices[i + 1]]["cost"])

                        for idx in [best_indices[i], best_indices[i + 1] if i + 1 < n_best else None]:
                            if idx is not None:
                                if self.flying_foxes[idx]["cost"] < self.best_value:
                                    self.best_solution = self.flying_foxes[idx]["position"].copy()
                                    self.best_value = self.flying_foxes[idx]["cost"]
                                if self.flying_foxes[idx]["cost"] > self.worst_value:
                                    self.worst_value = self.flying_foxes[idx]["cost"]

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration}: Function Evaluations = {self.func_count}, Best Cost = {self.best_value}")

            self.deltasO = self.deltasO_max - ((self.deltasO_max - self.deltasO_min) / self.max_evals) * self.func_count

        print("Optimization completed")
        return self.best_solution, self.best_value, self.history
