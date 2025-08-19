# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython
from itertools import combinations, chain
from libc.math cimport sin, cos, pi, sqrt
from cython cimport boundscheck, wraparound

# Define NumPy array types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

class AdaptiveDimensionSearch:
    def __init__(self, performance_function, int dim, bounds, int max_collocation_points=1000,
                 double epsilon_c=0.01, double gamma_c=0.01, double eta_c=0.01, int mcs_samples=10000):
        """
        Initialize the Adaptive Dimension Decomposition and Reselection (ADDR) algorithm.

        Parameters:
        - performance_function: Function g(x) to analyze reliability for.
        - dim: Number of dimensions (random variables).
        - bounds: Array of shape (dim, 2) with (lower, upper) bounds for each dimension.
        - max_collocation_points: Maximum number of collocation points (M_max).
        - epsilon_c: Relative error threshold for ADATP interpolation.
        - gamma_c: Threshold for dimensional weight to identify important dimensions.
        - eta_c: Threshold for test error indicator to reselect important dimensions.
        - mcs_samples: Number of Monte Carlo samples for reliability estimation.
        """
        self.performance_function = performance_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.max_collocation_points = max_collocation_points
        self.epsilon_c = epsilon_c
        self.gamma_c = gamma_c
        self.eta_c = eta_c
        self.mcs_samples = mcs_samples

        self.reference_point = np.mean(self.bounds, axis=1)  # mu_x
        self.important_dims = set()  # Set S
        self.constructed_dims = set()  # Set C
        self.selected_dims = set()  # Set R
        self.component_functions = {}  # Dict to store g_u^C
        self.collocation_points = 0
        self.g_max = -np.inf
        self.g_min = np.inf
        self.best_value = np.inf
        self.best_solution = None
        self.history = []  # Store (iteration, point, value)

    @boundscheck(False)
    @wraparound(False)
    def evaluate_function(self, np.ndarray[DTYPE_t, ndim=1] x):
        """Evaluate the performance function, update g_max, g_min, and track best solution."""
        cdef double value = self.performance_function(x)
        self.g_max = max(self.g_max, value)
        self.g_min = min(self.g_min, value)
        # Track minimum for optimization
        if value < self.best_value:
            self.best_value = value
            self.best_solution = x.copy()
        return value

    @boundscheck(False)
    @wraparound(False)
    def adATP_interpolation(self, u, int max_level=3):
        """
        Construct component function g_u^C using ADATP method.

        Parameters:
        - u: Tuple of dimension indices.
        - max_level: Maximum interpolation level.

        Returns:
        - Interpolated function, number of collocation points used, list of collocation points.
        """
        cdef list local_points = []
        cdef np.ndarray[DTYPE_t, ndim=1] x
        cdef double value
        cdef int idx, m, i
        cdef double lb, ub, pt
        cdef np.ndarray[DTYPE_t, ndim=1] theta, cc_points
        cdef double g_u
        cdef tuple v
        cdef int k
        cdef int sign

        if not u:
            x = self.reference_point.copy()
            value = self.evaluate_function(x)
            self.collocation_points += 1
            local_points.append(x.copy())
            self.history.append((self.collocation_points, x.copy(), value))
            return lambda x: value, 1, local_points

        # Simplified ADATP: Use Clenshaw-Curtis grid with piecewise linear basis
        cdef list points = []
        cdef list values = []
        cdef list weights = []
        for level in range(1, max_level + 1):
            m = 2 ** level + 1  # Number of points per dimension
            for idx in u:
                # Clenshaw-Curtis points in [0, 1]
                theta = np.linspace(0, pi, m)
                cc_points = 0.5 * (1.0 - np.cos(theta))
                # Map to actual bounds
                lb = self.bounds[idx, 0]
                ub = self.bounds[idx, 1]
                cc_points = lb + (ub - lb) * cc_points
                for pt in cc_points:
                    x = self.reference_point.copy()
                    x[idx] = pt
                    # Compute g_u^C using Cut-HDMR (Eq. 15)
                    g_u = 0.0
                    for v in chain.from_iterable(combinations(u, r) for r in range(len(u) + 1)):
                        x_v = x.copy()
                        for k in u:
                            if k not in v:
                                x_v[k] = self.reference_point[k]
                        sign = 1 if (len(u) - len(v)) % 2 == 0 else -1
                        g_u += sign * self.evaluate_function(x_v)
                    points.append(x[u])
                    values.append(g_u)
                    weights.append(1.0 / m)  # Simplified weight
                    self.collocation_points += 1
                    local_points.append(x.copy())
                    self.history.append((self.collocation_points, x.copy(), g_u))
                    if self.collocation_points >= self.max_collocation_points:
                        break
                if self.collocation_points >= self.max_collocation_points:
                    break
            if self.collocation_points >= self.max_collocation_points:
                break

        # Piecewise linear interpolation
        def interpolate(x_u):
            if not u:
                return values[0]
            # Handle empty input for non-empty u
            if len(x_u) == 0:
                return np.mean(np.array(values, dtype=DTYPE))
            # Find nearest point (simplified for demo)
            x_u_array = np.array(x_u, dtype=DTYPE)
            if len(u) == 1:  # Single dimension
                points_array = np.array(points, dtype=DTYPE).reshape(-1, 1)
                x_u_array = x_u_array.reshape(1,) if x_u_array.shape == () else x_u_array
            else:  # Multiple dimensions
                points_array = np.array(points, dtype=DTYPE)
            distances = np.sum((points_array - x_u_array) ** 2, axis=1)
            idx = np.argmin(distances)
            return values[idx]

        return interpolate, len(points), local_points

    @boundscheck(False)
    @wraparound(False)
    def compute_dimensional_weight(self, u, func):
        """
        Compute dimensional weight gamma_u (Eq. 23).

        Parameters:
        - u: Tuple of dimension indices.
        - func: Component function g_u^C.

        Returns:
        - Dimensional weight.
        """
        if self.g_max == self.g_min:
            return 0.0
        cdef double weight = 0.0
        cdef int samples = 100
        cdef int i, j
        cdef tuple v
        cdef int sign
        cdef np.ndarray[DTYPE_t, ndim=1] x, x_mapped
        cdef double value

        for v in chain.from_iterable(combinations(u, r) for r in range(len(u) + 1)):
            sign = 1 if (len(u) - len(v)) % 2 == 0 else -1
            for _ in range(samples):
                x = np.random.uniform(0, 1, len(v))
                # Map to bounds
                if v:
                    x_mapped = np.array([self.bounds[i, 0] + (self.bounds[i, 1] - self.bounds[i, 0]) * x[j]
                                         for j, i in enumerate(v)], dtype=DTYPE)
                    value = func(x_mapped)
                else:
                    value = func([])
                weight += sign * value / samples
        return weight / (self.g_max - self.g_min)

    @boundscheck(False)
    @wraparound(False)
    def compute_test_error_indicator(self, u):
        """
        Compute test error indicator eta_u (Eq. 25).

        Parameters:
        - u: Tuple of dimension indices.

        Returns:
        - Test error indicator.
        """
        if self.g_max == self.g_min:
            return 0.0
        cdef double eta = 0.0
        cdef np.ndarray[DTYPE_t, ndim=1] x_v
        cdef int i, k
        cdef tuple v
        cdef int sign

        for v in chain.from_iterable(combinations(u, r) for r in range(len(u) + 1)):
            x_v = self.reference_point.copy()
            for i in v:
                x_v[i] = self.bounds[i, 1]
            for k in range(self.dim):
                if k not in v:
                    x_v[k] = self.reference_point[k]
            sign = 1 if (len(u) - len(v)) % 2 == 0 else -1
            eta += sign * self.evaluate_function(x_v)
        return eta / (self.g_max - self.g_min)

    def select_dimensions(self, int order):
        """
        Select potentially important dimensions for order p (Eq. 24).

        Parameters:
        - order: Current decomposition order.

        Returns:
        - Set of selected dimension indices.
        """
        selected = set()
        possible_dims = list(combinations(range(self.dim), order))
        for u in possible_dims:
            admissible = True
            for v in chain.from_iterable(combinations(u, r) for r in range(1, len(u))):
                if tuple(sorted(v)) not in self.important_dims:
                    admissible = False
                    break
            if admissible:
                selected.add(tuple(sorted(u)))
        return selected

    def optimize(self):
        """
        Run the Adaptive Dimension Decomposition and Reselection algorithm.

        Returns:
        - Best solution, best value, history.
        """
        cdef int order = 1
        self.important_dims = set()
        self.constructed_dims = set()
        self.selected_dims = set()
        self.history = []

        # Zeroth-order
        func, points, local_points = self.adATP_interpolation(())
        self.component_functions[()] = func
        self.constructed_dims.add(())
        self.collocation_points = points
        weight = self.compute_dimensional_weight((), func)

        # First-order
        for i in range(self.dim):
            u = (i,)
            func, points, local_points = self.adATP_interpolation(u)
            self.component_functions[u] = func
            self.constructed_dims.add(u)
            weight = self.compute_dimensional_weight(u, func)
            if weight >= self.gamma_c:
                self.important_dims.add(u)
            if self.collocation_points >= self.max_collocation_points:
                break

        # Main loop
        while self.collocation_points < self.max_collocation_points:
            order += 1
            self.selected_dims = self.select_dimensions(order)
            if not self.selected_dims:
                break

            while self.selected_dims and self.collocation_points < self.max_collocation_points:
                u = self.selected_dims.pop()
                eta = self.compute_test_error_indicator(u)
                if eta >= self.eta_c:
                    func, points, local_points = self.adATP_interpolation(u)
                    self.component_functions[u] = func
                    self.constructed_dims.add(u)
                    weight = self.compute_dimensional_weight(u, func)
                    if weight >= self.gamma_c:
                        self.important_dims.add(u)

        print(f"Collocation Points Used: {self.collocation_points}")
        return self.best_solution, self.best_value, self.history
