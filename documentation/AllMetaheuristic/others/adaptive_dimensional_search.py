import numpy as np
from itertools import combinations, chain

class AdaptiveDimensionSearch:
    def __init__(self, performance_function, dim, bounds, max_collocation_points=1000,
                 epsilon_c=0.01, gamma_c=0.01, eta_c=0.01, mcs_samples=10000):
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
        self.bounds = np.array(bounds)
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
        self.g_max = -float("inf")
        self.g_min = float("inf")
        self.best_value = float("inf")
        self.best_solution = None
        self.history = []  # Store (iteration, point, value)

    def evaluate_function(self, x):
        """Evaluate the performance function, update g_max, g_min, and track best solution."""
        value = self.performance_function(x)
        self.g_max = max(self.g_max, value)
        self.g_min = min(self.g_min, value)
        # Track minimum for optimization
        if value < self.best_value:
            self.best_value = value
            self.best_solution = x.copy()
        return value

    def adATP_interpolation(self, u, max_level=3):
        """
        Construct component function g_u^C using ADATP method.

        Parameters:
        - u: Tuple of dimension indices.
        - max_level: Maximum interpolation level.

        Returns:
        - Interpolated function, number of collocation points used, list of collocation points.
        """
        local_points = []
        if not u:
            x = self.reference_point.copy()
            value = self.evaluate_function(x)
            self.collocation_points += 1
            local_points.append(x.copy())
            self.history.append((self.collocation_points, x.copy(), value))
            return lambda x: value, 1, local_points

        # Simplified ADATP: Use Clenshaw-Curtis grid with piecewise linear basis
        points = []
        values = []
        weights = []
        for level in range(1, max_level + 1):
            m = 2 ** level + 1  # Number of points per dimension
            for idx in u:
                # Clenshaw-Curtis points in [0, 1]
                theta = np.linspace(0, np.pi, m)
                cc_points = 0.5 * (1 - np.cos(theta))
                # Map to actual bounds
                lb, ub = self.bounds[idx]
                cc_points = lb + (ub - lb) * cc_points
                for pt in cc_points:
                    x = self.reference_point.copy()
                    x[idx] = pt
                    # Compute g_u^C using Cut-HDMR (Eq. 15)
                    g_u = 0
                    for v in chain.from_iterable(combinations(u, r) for r in range(len(u) + 1)):
                        x_v = x.copy()
                        for k in u:
                            if k not in v:
                                x_v[k] = self.reference_point[k]
                        sign = (-1) ** (len(u) - len(v))
                        g_u += sign * self.evaluate_function(x_v)
                    points.append(x[u])
                    values.append(g_u)
                    weights.append(1 / m)  # Simplified weight
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
                return np.mean(values)  # Return average of values for empty input
            # Find nearest point (simplified for demo)
            x_u = np.array(x_u)
            if len(u) == 1:  # Single dimension
                points_array = np.array(points).reshape(-1, 1)
                x_u = x_u.reshape(1,) if x_u.shape == () else x_u
            else:  # Multiple dimensions
                points_array = np.array(points)
            distances = np.sum((points_array - x_u) ** 2, axis=1)
            idx = np.argmin(distances)
            return values[idx]

        return interpolate, len(points), local_points

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
            return 0
        weight = 0
        samples = 100
        for v in chain.from_iterable(combinations(u, r) for r in range(len(u) + 1)):
            sign = (-1) ** (len(u) - len(v))
            for _ in range(samples):
                x = np.random.uniform(0, 1, len(v))
                x_mapped = np.array([self.bounds[i][0] + (self.bounds[i][1] - self.bounds[i][0]) * xi
                                     for i, xi in zip(v, x)] if v else [])
                value = func(x_mapped) if v else func([])
                weight += sign * value / samples
        return weight / (self.g_max - self.g_min)

    def compute_test_error_indicator(self, u):
        """
        Compute test error indicator eta_u (Eq. 25).

        Parameters:
        - u: Tuple of dimension indices.

        Returns:
        - Test error indicator.
        """
        if self.g_max == self.g_min:
            return 0
        eta = 0
        for v in chain.from_iterable(combinations(u, r) for r in range(len(u) + 1)):
            x_v = self.reference_point.copy()
            for i in v:
                x_v[i] = self.bounds[i][1]
            for i in range(self.dim):
                if i not in v:
                    x_v[i] = self.reference_point[i]
            sign = (-1) ** (len(u) - len(v))
            eta += sign * self.evaluate_function(x_v)
        return eta / (self.g_max - self.g_min)

    def select_dimensions(self, order):
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
        order = 1
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
