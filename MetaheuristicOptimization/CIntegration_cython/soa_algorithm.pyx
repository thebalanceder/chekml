import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log, sqrt
from libc.stdlib cimport rand, RAND_MAX
from posix.time cimport clock_gettime, timespec, CLOCK_PROCESS_CPUTIME_ID

# Define numpy types
cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double random():
    """Generate a random number between 0 and 1."""
    return rand() / <double>RAND_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class SeekerOptimizationAlgorithm:
    cdef object objective_function
    cdef int dim
    cdef cnp.ndarray bounds
    cdef int population_size
    cdef int max_fes
    cdef int max_gens
    cdef int num_regions
    cdef double mu_max
    cdef double mu_min
    cdef double w_max
    cdef double w_min
    cdef cnp.ndarray population
    cdef cnp.ndarray fitness
    cdef cnp.ndarray pbest_s
    cdef cnp.ndarray pbest_fun
    cdef cnp.ndarray gbest_s
    cdef double gbest_fun
    cdef cnp.ndarray lbest_s
    cdef cnp.ndarray lbest_fun
    cdef public list history  # Explicitly declare history as public
    cdef cnp.ndarray start_reg
    cdef cnp.ndarray end_reg
    cdef cnp.ndarray size_reg
    cdef cnp.ndarray rmax
    cdef cnp.ndarray rmin

    def __init__(self, objective_function, int dim=3, bounds=None, population_size=None, int max_fes=9000, 
                 int num_regions=3, double mu_max=0.99, double mu_min=0.0111, double w_max=0.9, double w_min=0.4):
        """
        Initialize the Seeker Optimization Algorithm (SOA).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: List of tuples [(min, max), ...] for each dimension.
        - population_size: Number of seekers (solutions). If None, computed based on dim.
        - max_fes: Maximum number of function evaluations.
        - num_regions: Number of sub-regions for population division.
        - mu_max, mu_min: Maximum and minimum mutation rates.
        - w_max, w_min: Maximum and minimum inertia weights.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.double) if bounds else np.array([[0.2, 1], [22, 30], [100, 108]], dtype=np.double)
        self.population_size = population_size if population_size else int(3 * (10 + 2 * sqrt(dim)))
        self.max_fes = max_fes
        self.max_gens = max_fes // self.population_size
        self.num_regions = num_regions
        self.mu_max = mu_max
        self.mu_min = mu_min
        self.w_max = w_max
        self.w_min = w_min

        # Initialize regions
        self.start_reg = np.zeros(num_regions, dtype=np.int32)
        self.end_reg = np.zeros(num_regions, dtype=np.int32)
        self.size_reg = np.zeros(num_regions, dtype=np.int32)
        cdef int region
        for region in range(num_regions):
            self.start_reg[region] = int(region * self.population_size / num_regions)
            self.end_reg[region] = int((region + 1) * self.population_size / num_regions)
            self.size_reg[region] = self.end_reg[region] - self.start_reg[region]

        self.rmax = 0.5 * (self.bounds[:, 1] - self.bounds[:, 0])
        self.rmin = -self.rmax

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void initialize_population(self):
        """Generate initial population randomly within bounds."""
        self.population = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                            (self.population_size, self.dim)).astype(np.double)
        self.fitness = np.zeros(self.population_size, dtype=np.double)
        self.pbest_s = self.population.copy()
        self.pbest_fun = np.full(self.population_size, np.inf, dtype=np.double)
        self.lbest_s = np.zeros((self.num_regions, self.dim), dtype=np.double)
        self.lbest_fun = np.full(self.num_regions, np.inf, dtype=np.double)
        self.gbest_s = np.zeros(self.dim, dtype=np.double)
        self.gbest_fun = np.inf
        self.history = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int evaluate_population(self, int fes):
        """Evaluate fitness of the population and update best solutions."""
        cdef int s
        cdef double fitness_val
        for s in range(self.population_size):
            fitness_val = self.objective_function(self.population[s])
            self.fitness[s] = fitness_val
            fes += 1

            # Update personal best
            if fitness_val < self.pbest_fun[s]:
                self.pbest_fun[s] = fitness_val
                self.pbest_s[s] = self.population[s].copy()

            # Update global best
            if fitness_val < self.gbest_fun:
                self.gbest_fun = fitness_val
                self.gbest_s = self.population[s].copy()

            self.history.append((fes, self.gbest_s.copy(), self.gbest_fun))
        return fes

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_region_bests(self):
        """Update local best solutions for each region."""
        cdef int region, best_idx
        cdef double best_fun
        for region in range(self.num_regions):
            region_indices = range(self.start_reg[region], self.end_reg[region])
            best_idx = self.start_reg[region] + np.argmin(self.fitness[region_indices])
            best_fun = self.fitness[best_idx]
            if best_fun < self.lbest_fun[region]:
                self.lbest_fun[region] = best_fun
                self.lbest_s[region] = self.population[best_idx].copy()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple optimize(self, int runs=30):
        """Run the Seeker Optimization Algorithm for multiple runs."""
        cdef cnp.ndarray final_gbest_fun = np.zeros(runs, dtype=np.double)
        cdef cnp.ndarray final_gbest_s = np.zeros((runs, self.dim), dtype=np.double)
        cdef cnp.ndarray final_time = np.zeros(runs, dtype=np.double)
        cdef cnp.ndarray func_value_curve = np.full((runs, self.max_fes), np.inf, dtype=np.double)
        cdef list gbest_curve = [[] for _ in range(runs)]
        cdef cnp.ndarray time_curve = np.zeros((runs, self.max_fes), dtype=np.double)

        cdef int run, fes, gens, s, region, rand_en, idx_local, i, j
        cdef double weight, mu, temp_val, error_prev
        cdef cnp.ndarray temp_population, e_t_1, e_t_2, f_t_1, f_t_2
        cdef cnp.ndarray x_offspring, x_pdirect, x_ldirect1, x_ldirect2, x_tdirect
        cdef cnp.ndarray en_temp, mu_s, x_direct, sign_x, r_temp
        cdef cnp.ndarray sorted_indices, perm
        cdef cnp.ndarray num_pone, num_none, prob_pone, prob_none, prob_zeros, rand_roulette
        cdef cnp.ndarray flag_cross
        cdef double start_time
        cdef timespec ts
        cdef cnp.ndarray log_mu_s = np.zeros(self.dim, dtype=np.double)

        for run in range(runs):
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts)
            start_time = ts.tv_sec + ts.tv_nsec / 1e9
            self.initialize_population()
            fes = 0
            gens = 0
            error_prev = np.inf

            # Initial evaluation
            fes = self.evaluate_population(fes)
            self.update_region_bests()

            e_t_1 = self.population.copy()
            e_t_2 = self.population.copy()
            f_t_1 = self.fitness.copy()
            f_t_2 = self.fitness.copy()

            while fes < self.max_fes:
                gens += 1
                weight = self.w_max - gens * (self.w_max - self.w_min) / self.max_gens

                temp_population = self.population.copy()
                for region in range(self.num_regions):
                    region_indices = range(self.start_reg[region], self.end_reg[region])
                    sorted_indices = np.argsort(self.fitness[region_indices])[::-1]
                    sorted_indices = np.array([i + self.start_reg[region] for i in sorted_indices], dtype=np.int32)

                    rand_en = max(1, <int>(random() * (self.size_reg[region] - 1) + 1))
                    en_temp = weight * np.abs(self.population[sorted_indices[-1]] - 
                                             self.population[sorted_indices[rand_en]])

                    mu = self.mu_max - gens * (self.mu_max - self.mu_min) / self.max_gens

                    for s in range(self.start_reg[region], self.end_reg[region]):
                        idx_local = np.where(sorted_indices == s)[0][0]
                        mu_s = mu + (1 - mu) * np.random.rand(self.dim)

                        x_pdirect = self.pbest_s[s] - self.population[s]
                        x_ldirect1 = self.lbest_s[region] - self.population[s] if self.lbest_fun[region] < self.fitness[s] else np.zeros(self.dim, dtype=np.double)
                        x_ldirect2 = self.population[sorted_indices[-1]] - self.population[s] if self.fitness[sorted_indices[-1]] < self.fitness[s] else np.zeros(self.dim, dtype=np.double)

                        # Temporal direction
                        f_values = np.array([f_t_2[s], f_t_1[s], self.fitness[s]], dtype=np.double)
                        e_values = np.array([e_t_2[s], e_t_1[s], self.population[s]], dtype=np.double)
                        order_idx = np.argsort(f_values)
                        x_tdirect = e_values[order_idx[0]] - e_values[order_idx[-1]]

                        flag_direct = np.array([1, 1, 1 if self.lbest_fun[region] < self.fitness[s] else 0, 
                                               1 if self.fitness[sorted_indices[-1]] < self.fitness[s] else 0], dtype=np.int32)
                        x_signs = np.array([np.sign(x_tdirect), np.sign(x_pdirect), 
                                           np.sign(x_ldirect1), np.sign(x_ldirect2)], dtype=np.double)
                        
                        select_sign = np.where(flag_direct > 0)[0]
                        num_sign = len(select_sign)
                        num_pone = np.zeros(self.dim, dtype=np.double)
                        num_none = np.zeros(self.dim, dtype=np.double)
                        
                        for i in select_sign:
                            num_pone += (np.abs(x_signs[i]) + x_signs[i]) / 2
                            num_none += (np.abs(x_signs[i]) - x_signs[i]) / 2
                        
                        num_zeros = num_sign - (num_pone + num_none)
                        prob_pone = num_pone / num_sign
                        prob_none = (num_pone + num_none) / num_sign
                        prob_zeros = np.ones(self.dim, dtype=np.double)
                        rand_roulette = np.random.rand(self.dim)
                        
                        x_direct = np.where(rand_roulette <= prob_pone, 1.0,
                                          np.where(rand_roulette <= prob_none, -1.0, 0.0))
                        
                        sign_x = np.sign(x_direct)
                        sign_x = np.where(self.population[s] > self.bounds[:, 1], -1.0, sign_x)
                        sign_x = np.where(self.population[s] < self.bounds[:, 0], 1.0, sign_x)
                        
                        zero_idx = np.where(sign_x == 0)[0]
                        if len(zero_idx) > 0:
                            sign_x[zero_idx] = np.random.choice([1.0, -1.0], size=len(zero_idx))
                        
                        # Compute log element-wise
                        for j in range(self.dim):
                            log_mu_s[j] = -2 * log(mu_s[j]) if mu_s[j] > 0 else 0
                        r_temp = sign_x * (en_temp * (log_mu_s ** 0.5))
                        r_temp = np.clip(r_temp, self.rmin, self.rmax)
                        
                        x_offspring = self.population[s] + r_temp
                        x_offspring = np.clip(x_offspring, self.bounds[:, 0], self.bounds[:, 1])
                        temp_population[s] = x_offspring

                    # Inter-region communication
                    si = 0
                    for region1 in range(self.num_regions):
                        if region1 != region:
                            si += 1
                            flag_cross = np.random.rand(self.dim) < 0.5
                            temp_population[sorted_indices[si]] = np.where(flag_cross, 
                                                                          self.lbest_s[region1], 
                                                                          temp_population[sorted_indices[si]])

                # Evaluate offspring
                for s in range(self.population_size):
                    x_offspring = temp_population[s]
                    temp_val = self.objective_function(x_offspring)
                    fes += 1

                    e_t_2[s] = e_t_1[s].copy()
                    e_t_1[s] = self.population[s].copy()
                    f_t_2[s] = f_t_1[s]
                    f_t_1[s] = self.fitness[s]
                    
                    self.fitness[s] = temp_val
                    self.population[s] = x_offspring.copy()

                    if temp_val < self.pbest_fun[s]:
                        self.pbest_fun[s] = temp_val
                        self.pbest_s[s] = x_offspring.copy()

                    if temp_val < self.gbest_fun:
                        self.gbest_fun = temp_val
                        self.gbest_s = x_offspring.copy()

                    func_value_curve[run, min(fes-1, self.max_fes-1)] = self.gbest_fun
                    gbest_curve[run].append((fes, self.gbest_s.copy()))
                    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts)
                    time_curve[run, min(fes-1, self.max_fes-1)] = (ts.tv_sec + ts.tv_nsec / 1e9) - start_time

                # Shuffle if no improvement or every generation
                if error_prev <= self.gbest_fun or gens % 1 == 0:
                    perm = np.random.permutation(self.population_size)
                    self.population = self.population[perm]
                    self.fitness = self.fitness[perm]
                    e_t_1 = e_t_1[perm]
                    e_t_2 = e_t_2[perm]
                    f_t_1 = f_t_1[perm]
                    f_t_2 = f_t_2[perm]
                    self.pbest_s = self.pbest_s[perm]
                    self.pbest_fun = self.pbest_fun[perm]

                error_prev = self.gbest_fun
                self.update_region_bests()

            final_gbest_fun[run] = self.gbest_fun
            final_gbest_s[run] = self.gbest_s.copy()
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts)
            final_time[run] = (ts.tv_sec + ts.tv_nsec / 1e9) - start_time

        return final_gbest_s, final_gbest_fun, func_value_curve, gbest_curve, time_curve
