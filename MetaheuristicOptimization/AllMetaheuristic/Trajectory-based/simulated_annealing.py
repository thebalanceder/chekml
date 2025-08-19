import numpy as np

class SimulatedAnnealing:
    def __init__(self, objective_function, dim=2, bounds=None,
                 init_temp=1.0, stop_temp=1e-8, max_tries=300,
                 max_success=20, max_consec_rej=1000, cool_schedule=None,
                 generator=None, stop_val=-np.inf, verbosity=1):
        """
        Initialize Simulated Annealing (SA).

        Parameters:
        - objective_function: Function to minimize.
        - dim: Number of dimensions.
        - bounds: List of (lower, upper) tuples for each dimension.
        - init_temp: Initial temperature.
        - stop_temp: Temperature at which to stop.
        - max_tries: Maximum number of tries per temperature.
        - max_success: Maximum number of accepted solutions per temperature.
        - max_consec_rej: Maximum number of consecutive rejections.
        - cool_schedule: Function to update the temperature.
        - generator: Function to generate new solution candidates.
        - stop_val: Objective function value that triggers early stopping.
        - verbosity: Level of logging (0: none, 1: final, 2: detailed).
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = bounds if bounds else [(-5, 5)] * dim
        self.init_temp = init_temp
        self.stop_temp = stop_temp
        self.max_tries = max_tries
        self.max_success = max_success
        self.max_consec_rej = max_consec_rej
        self.cool_schedule = cool_schedule or (lambda T: 0.8 * T)
        self.generator = generator or self._default_generator
        self.stop_val = stop_val
        self.verbosity = verbosity

        self.lower_bounds = np.array([b[0] for b in self.bounds])
        self.upper_bounds = np.array([b[1] for b in self.bounds])

        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    def _default_generator(self, x):
        perturbation = np.zeros_like(x)
        idx = np.random.randint(0, len(x))
        perturbation[idx] = np.random.randn() / 100
        return np.clip(x + perturbation, self.lower_bounds, self.upper_bounds)

    def _construct_initial_solution(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds)

    def optimize(self):
        T = self.init_temp
        k = 1  # Boltzmann constant
        parent = self._construct_initial_solution()
        old_energy = self.objective_function(parent)
        init_energy = old_energy
        consec_rejections = 0
        total_iterations = 0
        success = 0
        itry = 0

        if self.verbosity == 2:
            print(f"Initial T = {T:.5f}, loss = {old_energy:.5f}")

        while True:
            itry += 1
            current = parent.copy()

            if itry >= self.max_tries or success >= self.max_success:
                if T < self.stop_temp or consec_rejections >= self.max_consec_rej:
                    break
                T = self.cool_schedule(T)
                if self.verbosity == 2:
                    print(f"T = {T:.5f}, loss = {old_energy:.5f}")
                total_iterations += itry
                itry = 0
                success = 0

            new_param = self.generator(current)
            new_energy = self.objective_function(new_param)

            if new_energy < self.stop_val:
                parent = new_param
                old_energy = new_energy
                break

            delta = old_energy - new_energy

            if delta > 1e-6:
                parent = new_param
                old_energy = new_energy
                success += 1
                consec_rejections = 0
            else:
                if np.random.rand() < np.exp(delta / (k * T)):
                    parent = new_param
                    old_energy = new_energy
                    success += 1
                else:
                    consec_rejections += 1

        self.best_solution = parent
        self.best_fitness = old_energy
        self.history.append((total_iterations + itry, self.best_solution.copy()))

        if self.verbosity >= 1:
            print(f"Final Fitness = {self.best_fitness:.5f} after {total_iterations + itry} iterations")

        return self.best_solution, self.best_fitness, self.history

