# simulated_annealing.pyx

# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp

cdef class SimulatedAnnealing:
    cdef public object objective_function
    cdef public int dim, max_tries, max_success, max_consec_rej, verbosity
    cdef public double init_temp, stop_temp, stop_val
    cdef public object cool_schedule, generator, bounds
    cdef public np.ndarray lower_bounds, upper_bounds
    cdef public np.ndarray best_solution
    cdef public double best_fitness
    cdef public list history

    def __init__(self, object objective_function, int dim=2, bounds=None,
                 double init_temp=1.0, double stop_temp=1e-8,
                 int max_tries=300, int max_success=20, int max_consec_rej=1000,
                 object cool_schedule=None, object generator=None,
                 double stop_val=-1e308, int verbosity=1):

        self.objective_function = objective_function
        self.dim = dim
        self.bounds = bounds if bounds is not None else [(-5, 5)] * dim
        self.init_temp = init_temp
        self.stop_temp = stop_temp
        self.max_tries = max_tries
        self.max_success = max_success
        self.max_consec_rej = max_consec_rej
        self.cool_schedule = cool_schedule if cool_schedule is not None else (lambda T: 0.8 * T)
        self.generator = generator  # Custom generator or default
        self.stop_val = stop_val
        self.verbosity = verbosity

        self.lower_bounds = np.array([b[0] for b in self.bounds], dtype=np.float64)
        self.upper_bounds = np.array([b[1] for b in self.bounds], dtype=np.float64)

        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    cpdef np.ndarray _default_generator(self, np.ndarray[np.float64_t, ndim=1] x):
        cdef np.ndarray[np.float64_t, ndim=1] perturbation = np.zeros_like(x)
        cdef int idx = np.random.randint(0, x.shape[0])
        perturbation[idx] = np.random.randn() / 100
        return np.clip(x + perturbation, self.lower_bounds, self.upper_bounds)

    cpdef np.ndarray _construct_initial_solution(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds)

    cpdef tuple optimize(self):
        cdef double T = self.init_temp
        cdef double k = 1.0
        cdef np.ndarray[np.float64_t, ndim=1] parent = self._construct_initial_solution()
        cdef double old_energy = self.objective_function(parent)
        cdef int consec_rejections = 0
        cdef int total_iterations = 0
        cdef int success = 0
        cdef int itry = 0

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

            if self.generator:
                new_param = self.generator(current)
            else:
                new_param = self._default_generator(current)

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
                if np.random.rand() < exp(delta / (k * T)):
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

