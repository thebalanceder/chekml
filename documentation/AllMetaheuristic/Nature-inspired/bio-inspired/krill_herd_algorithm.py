import numpy as np

class KrillHerdOptimizer:
    def __init__(self, objective_function=None, dim=2, bounds=[(-5, 5)], population_size=25, max_iter=200, 
                 Vf=0.02, Dmax=0.005, Nmax=0.01, crossover_flag=True):
        """
        Initialize the Krill Herd Optimizer.

        Parameters:
        - objective_function: Function to optimize (default: Ackley function).
        - dim: Number of dimensions (variables).
        - bounds: List of (lower, upper) bounds for each dimension, e.g., [(-5, 5), (-5, 5)],
                  or a single tuple, e.g., (-5, 5), to apply to all dimensions.
        - population_size: Number of krill (solutions).
        - max_iter: Maximum number of iterations.
        - Vf: Foraging speed.
        - Dmax: Maximum diffusion speed.
        - Nmax: Maximum induced motion speed.
        - crossover_flag: Enable crossover operation (True/False).
        """
        self.objective_function = objective_function if objective_function else self.ackley_function
        self.dim = dim
        # Handle bounds
        if isinstance(bounds, tuple):
            bounds = [bounds] * dim  # Single tuple applies to all dimensions
        elif len(bounds) != dim:
            raise ValueError(f"Length of bounds ({len(bounds)}) must match dim ({dim})")
        self.bounds = np.array(bounds)  # Shape: (dim, 2)
        if self.bounds.shape != (dim, 2):
            raise ValueError(f"Bounds must be a list of (low, high) tuples with shape ({dim}, 2)")
        self.population_size = population_size
        self.max_iter = max_iter
        self.Vf = Vf
        self.Dmax = Dmax
        self.Nmax = Nmax
        self.crossover_flag = crossover_flag

        self.krill_positions = None  # Population of krill positions (solutions)
        self.best_position = None
        self.best_value = float("inf")
        self.history = []
        
        # Initialize motion parameters
        self.N = np.zeros((self.dim, self.population_size))  # Induced motion
        self.F = np.zeros((self.dim, self.population_size))  # Foraging motion
        self.D = np.zeros(self.population_size)  # Physical diffusion
        self.Dt = np.mean(np.abs(self.bounds[:, 1] - self.bounds[:, 0])) / 2  # Scale factor

    def ackley_function(self, X):
        """Ackley function as the default objective function."""
        n = self.dim
        a = 20
        b = 0.2
        c = 2 * np.pi
        s1 = np.sum(X ** 2)
        s2 = np.sum(np.cos(c * X))
        return -a * np.exp(-b * np.sqrt(s1 / n)) - np.exp(s2 / n) + a + np.exp(1)

    def initialize_krill_positions(self):
        """Generate initial krill positions randomly."""
        low = self.bounds[:, 0]  # Shape: (dim,)
        high = self.bounds[:, 1]  # Shape: (dim,)
        # Generate random positions with shape (dim, population_size)
        self.krill_positions = np.random.uniform(
            low[:, np.newaxis], high[:, np.newaxis], (self.dim, self.population_size)
        )

    def evaluate_krill_positions(self):
        """Compute fitness values for the krill positions."""
        return np.array([self.objective_function(self.krill_positions[:, i]) 
                         for i in range(self.population_size)])

    def find_limits(self, positions, best):
        """Evolutionary Boundary Constraint Handling Scheme."""
        ns = positions.copy()
        for i in range(self.dim):
            lb, ub = self.bounds[i, 0], self.bounds[i, 1]
            below = ns[i, :] < lb
            above = ns[i, :] > ub
            A = np.random.rand()
            B = np.random.rand()
            ns[i, below] = A * lb + (1 - A) * best[i]
            ns[i, above] = B * ub + (1 - B) * best[i]
        return ns

    def optimize(self):
        """Run the Krill Herd Optimization Algorithm."""
        self.initialize_krill_positions()
        K = self.evaluate_krill_positions()  # Fitness of krill
        Kib = K.copy()  # Best fitness of each krill
        Xib = self.krill_positions.copy()  # Best position of each krill
        best_idx = np.argmin(K)
        self.best_position = self.krill_positions[:, best_idx].copy()
        self.best_value = K[best_idx]
        Kgb = [self.best_value]  # Global best fitness history

        for iteration in range(self.max_iter):
            # Virtual food location
            Sf = np.sum(self.krill_positions / K.reshape(1, -1), axis=1)
            Xf = Sf / np.sum(1 / K)  # Food location
            Xf = self.find_limits(Xf[:, np.newaxis], self.best_position)[:, 0]
            Kf = self.objective_function(Xf)

            # Update food location if better
            if iteration > 0 and Kf < self.history[-1][2]:
                Xf = self.history[-1][1]
                Kf = self.history[-1][2]

            Kw_Kgb = np.max(K) - self.best_value
            w = 0.1 + 0.8 * (1 - iteration / self.max_iter)  # Inertia weight

            for i in range(self.population_size):
                # Calculate distances
                Rf = Xf - self.krill_positions[:, i]
                Rgb = self.best_position - self.krill_positions[:, i]
                RR = self.krill_positions - self.krill_positions[:, i:i+1]
                R = np.sqrt(np.sum(RR * RR, axis=0))

                # Movement Induced
                alpha_b = 0
                if self.best_value < K[i]:
                    alpha_b = -2 * (1 + np.random.rand() * (iteration / self.max_iter)) * \
                              (self.best_value - K[i]) / Kw_Kgb / np.sqrt(np.sum(Rgb * Rgb)) * Rgb

                alpha_n = 0
                nn = 0
                ds = np.mean(R) / 5
                for n in range(self.population_size):
                    if R[n] < ds and n != i:
                        nn += 1
                        if nn <= 4 and K[i] != K[n]:
                            alpha_n -= (K[n] - K[i]) / Kw_Kgb / R[n] * RR[:, n]
                            if nn >= 4:
                                break

                self.N[:, i] = w * self.N[:, i] + self.Nmax * (alpha_b + alpha_n)

                # Foraging Motion
                Beta_f = 0
                if Kf < K[i]:
                    Beta_f = -2 * (1 - iteration / self.max_iter) * (Kf - K[i]) / \
                             Kw_Kgb / np.sqrt(np.sum(Rf * Rf)) * Rf

                Rib = Xib[:, i] - self.krill_positions[:, i]
                Beta_b = 0
                if Kib[i] < K[i]:
                    Beta_b = -(Kib[i] - K[i]) / Kw_Kgb / np.sqrt(np.sum(Rib * Rib)) * Rib

                self.F[:, i] = w * self.F[:, i] + self.Vf * (Beta_b + Beta_f)

                # Physical Diffusion
                self.D = self.Dmax * (1 - iteration / self.max_iter) * \
                         np.floor(np.random.rand() + (K[i] - self.best_value) / Kw_Kgb) * \
                         (2 * np.random.rand(self.dim) - 1)

                # Motion Process
                DX = self.Dt * (self.N[:, i] + self.F[:, i])

                # Crossover
                if self.crossover_flag:
                    C_rate = 0.8 + 0.2 * (K[i] - self.best_value) / Kw_Kgb
                    Cr = np.random.rand(self.dim) < C_rate
                    NK4Cr = int(np.round((self.population_size - 1) * np.random.rand()))
                    self.krill_positions[:, i] = self.krill_positions[:, NK4Cr] * (~Cr) + \
                                                 self.krill_positions[:, i] * Cr

                # Update position
                self.krill_positions[:, i] += DX
                self.krill_positions[:, i] = self.find_limits(
                    self.krill_positions[:, i:i+1], self.best_position
                )[:, 0]

                # Evaluate new position
                K[i] = self.objective_function(self.krill_positions[:, i])
                if K[i] < Kib[i]:
                    Kib[i] = K[i]
                    Xib[:, i] = self.krill_positions[:, i]

            # Update global best
            best_idx = np.argmin(K)
            if K[best_idx] < self.best_value:
                self.best_position = self.krill_positions[:, best_idx].copy()
                self.best_value = K[best_idx]

            Kgb.append(self.best_value)
            self.history.append((iteration, self.best_position.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_position, self.best_value, self.history

if __name__ == "__main__":
    # Example usage
    optimizer = KrillHerdOptimizer(dim=2, bounds=[(-5, 5), (-5, 5)], population_size=25, max_iter=200)
    best_position, best_value, history = optimizer.optimize()
    print(f"Best Position: {best_position}")
    print(f"Best Value: {best_value}")
