import numpy as np

class DujiangyanIrrigationOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=100, 
                 diversion_factor=0.3, flow_adjustment=0.2, water_density=1.35, fluid_distribution=0.46, 
                 centrifugal_resistance=1.2, bottleneck_ratio=0.68, elimination_ratio=0.23):
        """
        Initialize the DISO optimizer.

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of water flow paths (solutions).
        - max_iter: Maximum number of iterations.
        - diversion_factor: Controls exploration.
        - flow_adjustment: Controls exploitation.
        - water_density: Density parameter for centrifugal force.
        - fluid_distribution: Fluid distribution coefficient.
        - centrifugal_resistance: Resistance coefficient for spiral motion.
        - bottleneck_ratio: Ratio for local refinement phase.
        - elimination_ratio: Percentage of worst solutions replaced per iteration.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.diversion_factor = diversion_factor
        self.flow_adjustment = flow_adjustment
        self.water_density = water_density
        self.fluid_distribution = fluid_distribution
        self.centrifugal_resistance = centrifugal_resistance
        self.bottleneck_ratio = bottleneck_ratio
        self.elimination_ratio = elimination_ratio

        self.water_flows = None  # Population of water flow paths (solutions)
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_water_flows(self):
        """ Generate initial water flow paths randomly """
        self.water_flows = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                             (self.population_size, self.dim))

    def evaluate_water_flows(self):
        """ Compute fitness values for the water flow paths """
        return np.array([self.objective_function(flow) for flow in self.water_flows])

    def diversion_phase(self, index):
        """ Simulate the Fish Mouth Dividing Project (global search) """
        r1, r2 = np.random.rand(), np.random.rand()
        HRO, HRI = 1.2, 7.2
        HGO, HGI = 1.3, 0.82
        CFR = 9.435 * np.random.gamma(0.85, 2.5)  # Comprehensive riverbed roughness

        if r1 < 0.23:
            Vi1 = (HRO ** (2/3) * HGO ** (1/2)) / CFR * r1
        else:
            Vi1 = (HRI ** (2/3) * HGI ** (1/2)) / CFR * r2

        new_solution = (self.best_solution if self.best_solution is not None else self.water_flows[index]) + \
                       (self.best_solution - self.water_flows[index]) * np.random.rand(self.dim) * Vi1

        return np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

    def spiral_motion_update(self, X, X_best, t):
        """ 
        Simulate inner river flow with centrifugal & lateral pressure effects.
        
        - Applies spiral motion based on RCF & LP calculations.
        """
        num_solutions, num_dimensions = X.shape
        T = self.max_iter  # Maximum iterations

        # Compute centrifugal force (RCF)
        RCF = self.water_density * np.cos(90 * (t / T)) * np.sqrt(np.sum((X_best - X) ** 2, axis=1, keepdims=True))

        # Compute mean longitudinal velocity (MLV)
        fitness = self.evaluate_water_flows().reshape(-1, 1)
        MLV = np.mean(fitness)

        # Compute lateral pressure (LP)
        LP = (self.water_density * self.fluid_distribution * MLV ** 2) / self.centrifugal_resistance

        # Apply spiral motion condition
        mask = RCF < LP
        LB, UB = self.bounds[:, 0], self.bounds[:, 1]
        new_X = np.where(mask, 
                         X, 
                         (UB - LB) * np.random.rand(num_solutions, num_dimensions) + LB
                        )

        return new_X

    def local_development_phase(self, index):
        """ Simulate Baopingkou Project (local refinement) """
        r3, r4 = np.random.rand(), np.random.rand()
        HRI, HGI = 7.2, 0.82
        CFR = 9.435 * np.random.gamma(0.85, 2.5)

        if r3 < self.bottleneck_ratio:
            Vi2 = (HRI ** (2/3) * HGI ** (1/2)) / (2 * CFR) * r3
        else:
            Vi2 = (HRI ** (2/3) * HGI ** (1/2)) / (2 * CFR) * r4

        Improve2 = np.sign(self.best_value - self.evaluate_water_flows()[index]) * \
                   (self.best_solution - self.water_flows[index]) * np.random.rand(self.dim)

        new_solution = self.best_solution + (self.best_solution - self.water_flows[index]) * \
                       np.random.rand(self.dim) * Vi2 + Improve2

        return np.clip(new_solution, self.bounds[:, 0], self.bounds[:, 1])

    def elimination_phase(self):
        """ Simulate Feishayan Sediment Discharge (worst solution replacement) """
        fitness = self.evaluate_water_flows()
        worst_indices = np.argsort(fitness)[-int(self.elimination_ratio * self.population_size):]
        for i in worst_indices:
            self.water_flows[i] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)

    def optimize(self):
        """ Run the Dujiangyan Irrigation System Optimization """
        self.initialize_water_flows()
        for generation in range(self.max_iter):
            fitness = self.evaluate_water_flows()
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.water_flows[min_idx]
                self.best_value = fitness[min_idx]

            # Global search (Fish Mouth Dividing Project)
            for i in range(self.population_size):
                self.water_flows[i] = self.diversion_phase(i)

            # Spiral motion update (inner river flow)
            self.water_flows = self.spiral_motion_update(self.water_flows, self.best_solution, generation)

            # Local development (Baopingkou Project)
            for i in range(self.population_size):
                self.water_flows[i] = self.local_development_phase(i)

            # Elimination (Feishayan Sediment Discharge)
            self.elimination_phase()

            self.history.append((generation, self.best_solution.copy(), self.best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history