import numpy as np

class GreatWallConstructionOptimizer:
    def __init__(self, objective_function, dim, bounds, population_size=30, max_iter=500, runs=1):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)  # Bounds as [(low, high), ...]
        self.population_size = population_size
        self.max_iter = max_iter
        self.runs = runs

        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    def initialize_population(self):
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def evaluate_population(self, population):
        return np.array([self.objective_function(ind) for ind in population])

    def optimize(self):
        g = 9.8
        m = 3
        e = 0.1
        P = 9
        Q = 6
        Cmax = np.exp(3)
        Cmin = np.exp(2)

        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        best_overall = float("inf")
        best_solution_overall = None

        for run in range(self.runs):
            population = self.initialize_population()
            fitness = self.evaluate_population(population)

            sorted_indices = np.argsort(fitness)
            Worker1, Worker2, Worker3 = population[sorted_indices[:3]]
            Worker1_fit, Worker2_fit, Worker3_fit = fitness[sorted_indices[:3]]

            LNP = int(np.ceil(self.population_size * e))

            for t in range(1, self.max_iter + 1):
                C = Cmax - ((Cmax - Cmin) * t / self.max_iter)

                for i in range(self.population_size):
                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    if i < LNP:
                        F = (m * g * r1) / (P * Q * (1 + t))
                        new_pos = population[i] + F * np.sign(np.random.randn(self.dim)) * C
                    else:
                        influence = (Worker1 + Worker2 + Worker3) / 3 - population[i]
                        new_pos = population[i] + r2 * influence * C

                    new_pos = np.clip(new_pos, lb, ub)
                    new_fit = self.objective_function(new_pos)

                    if new_fit < fitness[i]:
                        population[i] = new_pos
                        fitness[i] = new_fit

                        if new_fit < Worker1_fit:
                            Worker3, Worker3_fit = Worker2.copy(), Worker2_fit
                            Worker2, Worker2_fit = Worker1.copy(), Worker1_fit
                            Worker1, Worker1_fit = new_pos.copy(), new_fit

                if Worker1_fit < best_overall:
                    best_overall = Worker1_fit
                    best_solution_overall = Worker1.copy()

                self.history.append((t, Worker1_fit))
                print(f"Iteration {t}: Best Value = {Worker1_fit}")

        self.best_solution = best_solution_overall
        self.best_value = best_overall
        return self.best_solution, self.best_value, self.history


# Example usage
if __name__ == '__main__':
    def sphere(x):
        return np.sum(x**2)

    bounds = [(-100, 100)] * 30
    gwca = GreatWallConstructionOptimizer(sphere, dim=30, bounds=np.array(bounds))
    best_sol, best_val, history = gwca.optimize()
    print("Best Fitness:", best_val)
    print("Best Solution:", best_sol)

