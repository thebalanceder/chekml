# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport round, exp, fabs

cnp.import_array()

cdef class DynamicHuntingLeadershipOptimizer:
    cdef public:
        object objective_function
        int dim, population_size, max_iter, num_leaders
        str variant
        double tolerance
        cnp.ndarray bounds, search_agents, leaders_pos, leaders_fit
        cnp.ndarray best_solution
        double best_value
        list convergence_curve, num_leaders_history, position_history
        cnp.ndarray leader_history
        int current_num_leaders

    def __init__(self, object objective_function, int dim, object bounds, int population_size=30,
                 int max_iter=200, int num_leaders=30, str variant='V4', double tolerance=5):
        """
        Initialize the Dynamic Hunting Leadership Optimizer (DHL).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.num_leaders = num_leaders
        self.variant = variant.upper()
        self.tolerance = tolerance

        # Validate variant
        if self.variant not in ['V1', 'V2', 'V3', 'V4']:
            raise ValueError("Variant must be one of 'V1', 'V2', 'V3', 'V4'.")

        # Process bounds
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2 and not isinstance(bounds[0], (list, tuple)):
            self.bounds = np.array([[bounds[0], bounds[1]]] * dim, dtype=np.double)
        else:
            self.bounds = np.array(bounds, dtype=np.double)
            if self.bounds.shape[0] != dim:
                raise ValueError("Bounds must match the number of dimensions or be a single (lb, ub) pair.")

        self.best_value = float("inf")
        self.convergence_curve = []
        self.num_leaders_history = []
        self.position_history = []
        self.current_num_leaders = 0  # Initialized in optimize

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void initialize_population(self):
        """Initialize the positions of search agents and leaders."""
        self.leaders_pos = self._initialize(self.num_leaders, self.dim)
        self.leaders_fit = np.full(self.num_leaders, float("inf"), dtype=np.double)
        self.search_agents = self._initialize(self.population_size, self.dim)
        self.leader_history = np.zeros((self.num_leaders, self.max_iter), dtype=np.double)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=2] _initialize(self, int num_agents, int dim):
        """Helper function to initialize positions within bounds."""
        cdef cnp.ndarray[cnp.double_t, ndim=2] positions = np.zeros((num_agents, dim), dtype=np.double)
        cdef int i, j
        cdef double lb, ub
        for i in range(dim):
            lb = self.bounds[i, 0]
            ub = self.bounds[i, 1]
            for j in range(num_agents):
                positions[j, i] = lb + (ub - lb) * np.random.rand()
        return positions

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=1] check_bounds(self, cnp.ndarray[cnp.double_t, ndim=1] position):
        """Ensure position stays within bounds."""
        cdef int i
        cdef double lb, ub
        cdef cnp.ndarray[cnp.double_t, ndim=1] clipped = np.empty(self.dim, dtype=np.double)
        for i in range(self.dim):
            lb = self.bounds[i, 0]
            ub = self.bounds[i, 1]
            clipped[i] = max(lb, min(ub, position[i]))
        return clipped

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[cnp.double_t, ndim=1] evaluate_population(self):
        """Compute fitness values for the search agents."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness = np.empty(self.population_size, dtype=np.double)
        cdef int i
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.search_agents[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray[cnp.double_t, ndim=1] update_leaders(self, int iter_idx):
        """Update leader positions and fitness based on search agents' fitness."""
        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness = self.evaluate_population()
        cdef cnp.ndarray[cnp.double_t, ndim=1] leader_history = np.full(self.num_leaders, float("inf"), dtype=np.double) if iter_idx == 0 else self.leader_history[:, iter_idx-1]
        cdef int i, i_b = 0  # Initialize i_b to avoid uninitialized warning
        cdef double sa_fit
        for i in range(self.population_size):
            sa_fit = fitness[i]
            for i_b in range(self.current_num_leaders):
                if sa_fit < self.leaders_fit[i_b]:
                    self.leaders_fit[i_b] = sa_fit
                    self.leaders_pos[i_b] = self.search_agents[i].copy()
                    break
            leader_history[i_b] = self.leaders_fit[i_b]
        return leader_history

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_num_leaders(self, int iter_idx, cnp.ndarray[cnp.double_t, ndim=1] leader_history):
        """Dynamically adjust the number of leaders based on the variant."""
        cdef int n_L
        cdef double AA, tol_iter
        if self.variant == 'V1':
            AA = 1 - iter_idx / (self.max_iter / 2)
            if AA > 0:
                n_L = <int>round(3 + AA * (self.num_leaders - 3))
            else:
                AA = 1 - (iter_idx / self.max_iter)
                n_L = <int>round(1 + AA * 4)
        elif self.variant == 'V2':
            AA = 1 - iter_idx / self.max_iter
            n_L = <int>round(1 + AA * (self.num_leaders - 1))
        elif self.variant == 'V3':
            AA = -iter_idx * 10 / self.max_iter
            n_L = <int>round(self.num_leaders * exp(AA) + 1)
        else:  # V4
            tol_iter = self.max_iter * self.tolerance / 100
            n_L = self.current_num_leaders
            if iter_idx >= tol_iter + 1:
                if leader_history[n_L - 1] == float("inf"):
                    n_L -= 1
                elif fabs(leader_history[n_L - 1] - self.leader_history[n_L - 1, iter_idx - <int>tol_iter]) < 1e-5:
                    n_L -= 1

        n_L = max(1, min(n_L, self.num_leaders))
        self.current_num_leaders = n_L
        self.num_leaders_history.append(n_L)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_positions(self, int iter_idx):
        """Update search agents' positions based on leaders."""
        cdef double a = 2 - iter_idx * (2.0 / self.max_iter)
        cdef int i, j, i_x
        cdef double r1, r2, A1, C1, D_alpha, XX_sum
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_pos = np.empty(self.dim, dtype=np.double)
        cdef cnp.ndarray[cnp.double_t, ndim=1] XX = np.empty(self.current_num_leaders, dtype=np.double)

        for i in range(self.population_size):
            for j in range(self.dim):
                XX_sum = 0.0
                for i_x in range(self.current_num_leaders):
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = fabs(C1 * self.leaders_pos[i_x, j] - self.search_agents[i, j])
                    XX[i_x] = self.leaders_pos[i_x, j] - A1 * D_alpha
                    XX_sum += XX[i_x]
                new_pos[j] = XX_sum / self.current_num_leaders
            self.search_agents[i] = self.check_bounds(new_pos)

    def optimize(self):
        """Run the Dynamic Hunting Leadership Optimization."""
        self.initialize_population()
        self.current_num_leaders = self.num_leaders
        cdef cnp.ndarray[cnp.double_t, ndim=1] leader_history
        cdef int iter_idx

        for iter_idx in range(self.max_iter):
            leader_history = self.update_leaders(iter_idx)
            self.leader_history[:, iter_idx] = leader_history
            self.update_num_leaders(iter_idx, leader_history)
            self.update_positions(iter_idx)

            if self.leaders_fit[0] < self.best_value:
                self.best_value = self.leaders_fit[0]
                self.best_solution = self.leaders_pos[0].copy()
            self.position_history.append(self.best_solution.copy())  # Store clipped position

            self.convergence_curve.append(self.best_value)

            print(f"Iteration {iter_idx + 1}: Best Value = {self.best_value}, Num Leaders = {self.current_num_leaders}")

        return self.best_solution, self.best_value, self.convergence_curve, self.num_leaders_history, self.position_history

# Example usage
if __name__ == "__main__":
    def sphere_function(x):
        """Example objective function: Sphere function."""
        return np.sum(x ** 2)

    dim = 30
    bounds = [(-100, 100)] * dim
    optimizer = DynamicHuntingLeadershipOptimizer(
        objective_function=sphere_function,
        dim=dim,
        bounds=bounds,
        population_size=30,
        max_iter=200,
        num_leaders=30,
        variant='V4',
        tolerance=5
    )
    best_solution, best_value, convergence_curve, num_leaders_history, position_history = optimizer.optimize()
    print(f"Best Solution: {best_solution}")
    print(f"Best Value: {best_value}")
