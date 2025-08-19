# cython: language_level=3
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport exp, ceil, fabs

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class PartialReinforcementOptimizer:
    cdef public:
        object objective_function
        int dim
        cnp.ndarray bounds
        int population_size
        int max_evaluations
        double reinforcement_rate
        cnp.ndarray population
        cnp.ndarray schedules
        cnp.ndarray best_solution
        double best_value
        list history
        int evaluations

    def __init__(self, objective_function, int dim, bounds, int population_size=30, 
                 int max_evaluations=10000, double reinforcement_rate=0.7):
        """
        Initialize the Partial Reinforcement Optimizer (PRO).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Tuple of (lower, upper) bounds for each dimension.
        - population_size: Number of individuals (solutions).
        - max_evaluations: Maximum number of function evaluations.
        - reinforcement_rate: Reinforcement rate (RR) for schedule updates.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double)
        self.population_size = population_size
        self.max_evaluations = max_evaluations
        self.reinforcement_rate = reinforcement_rate
        self.population = None
        self.schedules = None
        self.best_solution = None
        self.best_value = float("inf")
        self.history = []
        self.evaluations = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void initialize_population(self):
        """ Generate initial population and schedules randomly """
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                           (self.population_size, self.dim))
        self.schedules = np.random.uniform(0.9, 1.0, (self.population_size, self.dim))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray evaluate_population(self):
        """ Compute fitness values for the population """
        cdef int i
        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness = np.empty(self.population_size, dtype=np.double)
        for i in range(self.population_size):
            fitness[i] = self.objective_function(self.population[i])
        return fitness

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray[cnp.intp_t, ndim=1] select_behaviors(self, int i, int current_eval):
        """ Select behaviors based on schedule priorities """
        cdef double tau = current_eval / <double>self.max_evaluations
        cdef double selection_rate = exp(-(1 - tau))
        cdef cnp.ndarray[cnp.intp_t, ndim=1] sorted_indices = np.argsort(self.schedules[i])[::-1]
        cdef int landa = <int>ceil(self.dim * np.random.rand() * selection_rate)
        return sorted_indices[:landa]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef cnp.ndarray[cnp.double_t, ndim=1] stimulate_behaviors(self, int i, 
                                                               cnp.ndarray[cnp.intp_t, ndim=1] selected_behaviors, 
                                                               int k, int current_eval):
        """ Stimulate selected behaviors to generate new solution """
        cdef double tau = current_eval / <double>self.max_evaluations
        cdef cnp.ndarray[cnp.double_t, ndim=1] stimulation = np.zeros(self.dim, dtype=np.double)
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_solution = self.population[i].copy()
        cdef double sf, rand_val
        cdef int j, idx
        cdef cnp.ndarray[cnp.uint8_t, ndim=1, cast=True] under_lb, over_ub

        rand_val = np.random.rand()
        if rand_val < 0.5:
            for j in range(selected_behaviors.shape[0]):
                idx = selected_behaviors[j]
                stimulation[idx] = self.best_solution[idx] - self.population[i, idx]
        else:
            for j in range(selected_behaviors.shape[0]):
                idx = selected_behaviors[j]
                stimulation[idx] = self.population[i, idx] - self.population[k, idx]

        # Calculate Stimulation Factor (SF)
        cdef double schedule_mean = 0.0
        cdef double schedule_max = 0.0
        for j in range(selected_behaviors.shape[0]):
            idx = selected_behaviors[j]
            schedule_mean += self.schedules[i, idx]
            schedule_max = max(schedule_max, fabs(self.schedules[i, idx]))
        schedule_mean /= selected_behaviors.shape[0] if selected_behaviors.shape[0] > 0 else 1.0
        sf = tau + np.random.rand() * (schedule_mean / schedule_max if schedule_max > 0 else 1.0)

        for j in range(selected_behaviors.shape[0]):
            idx = selected_behaviors[j]
            new_solution[idx] += sf * stimulation[idx]

        # Bound constraints control
        under_lb = new_solution < self.bounds[:, 0]
        over_ub = new_solution > self.bounds[:, 1]
        for j in range(self.dim):
            if under_lb[j]:
                new_solution[j] = self.bounds[j, 0] + np.random.rand() * (self.bounds[j, 1] - self.bounds[j, 0])
            if over_ub[j]:
                new_solution[j] = self.bounds[j, 0] + np.random.rand() * (self.bounds[j, 1] - self.bounds[j, 0])

        return new_solution

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void apply_reinforcement(self, int i, cnp.ndarray[cnp.intp_t, ndim=1] selected_behaviors, 
                                  cnp.ndarray[cnp.double_t, ndim=1] new_solution, double new_value):
        """ Apply positive or negative reinforcement based on response """
        cdef double current_value = self.objective_function(self.population[i])
        cdef int j, idx
        if new_value < current_value:
            # Positive Reinforcement
            for j in range(selected_behaviors.shape[0]):
                idx = selected_behaviors[j]
                self.schedules[i, idx] += self.schedules[i, idx] * (self.reinforcement_rate / 2)
            self.population[i] = new_solution
            if new_value < self.best_value:
                self.best_solution = new_solution.copy()
                self.best_value = new_value
        else:
            # Negative Reinforcement
            for j in range(selected_behaviors.shape[0]):
                idx = selected_behaviors[j]
                self.schedules[i, idx] -= self.schedules[i, idx] * self.reinforcement_rate

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void reschedule(self, int i):
        """ Reschedule if schedule variance is zero """
        cdef cnp.ndarray[cnp.double_t, ndim=1] sched = self.schedules[i]
        cdef double mean = np.mean(sched)
        cdef double variance = np.mean((sched - mean) ** 2)
        if variance == 0:
            self.schedules[i] = np.random.uniform(0.9, 1.0, self.dim)
            self.population[i] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
            print(f"Learner {i} is Rescheduled")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self):
        """ Run the Partial Reinforcement Optimization """
        self.initialize_population()
        self.best_solution = self.population[0].copy()
        self.best_value = self.objective_function(self.best_solution)

        cdef cnp.ndarray[cnp.double_t, ndim=1] fitness
        cdef cnp.ndarray[cnp.intp_t, ndim=1] sorted_indices
        cdef cnp.ndarray[cnp.intp_t, ndim=1] selected_behaviors
        cdef cnp.ndarray[cnp.double_t, ndim=1] new_solution
        cdef double new_value
        cdef int i, k

        while self.evaluations < self.max_evaluations:
            fitness = self.evaluate_population()
            sorted_indices = np.argsort(fitness)
            self.population = self.population[sorted_indices]
            self.schedules = self.schedules[sorted_indices]
            if fitness[sorted_indices[0]] < self.best_value:
                self.best_solution = self.population[0].copy()
                self.best_value = fitness[sorted_indices[0]]

            for i in range(self.population_size):
                # Select another individual for comparison
                k = self.population_size - 1
                if i < self.population_size - 1:
                    k = sorted_indices[np.random.randint(i + 1, self.population_size)]

                # Select behaviors
                selected_behaviors = self.select_behaviors(i, self.evaluations)

                # Stimulate behaviors
                new_solution = self.stimulate_behaviors(i, selected_behaviors, k, self.evaluations)

                # Evaluate new solution
                new_value = self.objective_function(new_solution)
                self.evaluations += 1

                # Apply reinforcement
                self.apply_reinforcement(i, selected_behaviors, new_solution, new_value)

                # Reschedule if necessary
                self.reschedule(i)

                # Store best cost for current evaluation
                self.history.append((self.evaluations, self.best_solution.copy(), self.best_value))
                
                # Check if max evaluations reached
                if self.evaluations >= self.max_evaluations:
                    break

            print(f"Iteration {self.evaluations}: Best Value = {self.best_value}")

        return self.best_solution, self.best_value, self.history
