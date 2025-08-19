# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, pi
from libc.stdlib cimport rand, srand
from libc.time cimport time
from sklearn.cluster import KMeans
cimport cython

np.import_array()

cdef class CuckooOptimizer:
    cdef public int dim, population_size, max_iter, min_eggs, max_eggs, max_cuckoos, knn_cluster_num
    cdef public double radius_coeff, motion_coeff, variance_threshold
    cdef np.ndarray bounds, cuckoos
    cdef object objective_function
    cdef object best_solution
    cdef double best_value
    cdef list history

    def __init__(self, objective_function, int dim, bounds, int population_size=5, int max_iter=100,
                 int min_eggs=2, int max_eggs=4, int max_cuckoos=10, double radius_coeff=5.0,
                 double motion_coeff=9.0, int knn_cluster_num=1, double variance_threshold=1e-13):
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_iter = max_iter
        self.min_eggs = min_eggs
        self.max_eggs = max_eggs
        self.max_cuckoos = max_cuckoos
        self.radius_coeff = radius_coeff
        self.motion_coeff = motion_coeff
        self.knn_cluster_num = knn_cluster_num
        self.variance_threshold = variance_threshold

        self.cuckoos = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []

    cpdef initialize_cuckoos(self):
        self.cuckoos = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                         (self.population_size, self.dim)).astype(np.float64)

    cpdef evaluate_cuckoos(self, np.ndarray[np.float64_t, ndim=2] positions):
        return np.apply_along_axis(self.objective_function, 1, positions)

    cpdef lay_eggs(self):
        cdef int i, d, n_eggs
        cdef np.ndarray[np.float64_t, ndim=1] bound_range = self.bounds[:, 1] - self.bounds[:, 0]
        num_eggs = np.random.randint(self.min_eggs, self.max_eggs + 1, self.population_size)
        total_eggs = np.sum(num_eggs)
        radius_factors = (num_eggs / total_eggs) * self.radius_coeff

        egg_positions = []
        for i in range(self.population_size):
            n_eggs = num_eggs[i]
            center = self.cuckoos[i]
            random_scalars = np.random.rand(n_eggs)
            radii = radius_factors[i] * random_scalars[:, np.newaxis] * bound_range
            angles = np.linspace(0, 2 * np.pi, n_eggs)

            rand_signs = (-1) ** np.random.randint(1, 3, n_eggs)
            adding_values = np.zeros((n_eggs, self.dim))
            for d in range(self.dim):
                adding_values[:, d] = rand_signs * radii[:, d] * np.cos(angles) + radii[:, d] * np.sin(angles)

            new_positions = center + adding_values
            egg_positions.append(np.clip(new_positions, self.bounds[:, 0], self.bounds[:, 1]))

        all_eggs = np.vstack(egg_positions)
        return np.unique(all_eggs, axis=0)

    cpdef select_best_cuckoos(self, np.ndarray[np.float64_t, ndim=2] positions, fitness):
        if len(positions) <= self.max_cuckoos:
            return positions, fitness
        sorted_indices = np.argsort(fitness)[:self.max_cuckoos]
        return positions[sorted_indices], fitness[sorted_indices]

    cpdef cluster_and_migrate(self, np.ndarray[np.float64_t, ndim=2] positions):
        if np.sum(np.var(positions, axis=0)) < self.variance_threshold:
            return positions, True

        kmeans = KMeans(n_clusters=self.knn_cluster_num, n_init=1, random_state=0).fit(positions)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        clusters = [[] for _ in range(self.knn_cluster_num)]
        cluster_fitness = [[] for _ in range(self.knn_cluster_num)]
        for i, label in enumerate(labels):
            clusters[label].append(positions[i])
            cluster_fitness[label].append(self.objective_function(positions[i]))

        for i in range(self.knn_cluster_num):
            clusters[i] = np.array(clusters[i]) if clusters[i] else np.empty((0, self.dim))
            cluster_fitness[i] = np.array(cluster_fitness[i]) if cluster_fitness[i] else np.array([])

        mean_fitness = [np.mean(f) if len(f) > 0 else float("inf") for f in cluster_fitness]
        best_cluster_idx = np.argmin(mean_fitness)
        best_cluster = clusters[best_cluster_idx]
        if len(best_cluster) == 0:
            return positions, False

        best_point_idx = np.argmin(cluster_fitness[best_cluster_idx])
        goal_point = best_cluster[best_point_idx]

        new_positions = positions + self.motion_coeff * np.random.rand(len(positions), self.dim) * (
            goal_point - positions
        )
        new_positions = np.clip(new_positions, self.bounds[:, 0], self.bounds[:, 1])
        return new_positions, False

    cpdef optimize(self):
        self.initialize_cuckoos()
        cdef int iteration, min_idx
        cdef double current_best

        for iteration in range(self.max_iter):
            fitness = self.evaluate_cuckoos(self.cuckoos)
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.cuckoos[min_idx].copy()
                self.best_value = fitness[min_idx]

            egg_positions = self.lay_eggs()
            egg_fitness = self.evaluate_cuckoos(egg_positions)

            all_positions = np.vstack([self.cuckoos, egg_positions])
            all_fitness = np.concatenate([fitness, egg_fitness])

            self.cuckoos, fitness = self.select_best_cuckoos(all_positions, all_fitness)
            self.population_size = len(self.cuckoos)

            self.cuckoos, stop = self.cluster_and_migrate(self.cuckoos)
            if stop:
                break

            best_fitness = self.evaluate_cuckoos(self.cuckoos)
            if np.min(best_fitness) > self.best_value:
                self.cuckoos[-1] = self.best_solution.copy()
            else:
                min_idx = np.argmin(best_fitness)
                if best_fitness[min_idx] < self.best_value:
                    self.best_solution = self.cuckoos[min_idx].copy()
                    self.best_value = best_fitness[min_idx]

            if len(self.cuckoos) > 1:
                randomized_best = self.best_solution * np.random.rand(self.dim)
                randomized_best = np.clip(randomized_best, self.bounds[:, 0], self.bounds[:, 1])
                self.cuckoos[-2] = randomized_best

            self.history.append((iteration, self.best_solution.copy(), self.best_value))
            print(f"Iteration {iteration + 1}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history

