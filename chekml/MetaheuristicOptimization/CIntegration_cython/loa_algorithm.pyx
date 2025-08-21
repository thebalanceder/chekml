# cython: language_level=3
# cython: boundscheck=True
# cython: wraparound=True

# boundscheck and wraparound temporarily enabled for debugging

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport cos, sin, tan, ceil, sqrt
from libc.stdlib cimport rand, RAND_MAX

# Define types for NumPy arrays
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# Random number generator helper
cdef inline double rand_uniform():
    return rand() / <double>RAND_MAX

cdef class LionOptimizationAlgorithm:
    cdef:
        object objective_function
        int dim
        double[:, :] bounds
        int population_size
        int max_iter
        double nomad_ratio
        int pride_size
        double female_ratio
        double roaming_ratio
        double mating_ratio
        double mutation_prob
        double immigration_ratio
        double[:, :] lions
        double[:, :] best_positions
        double[:] best_fitness
        double[:] global_best_solution
        double global_best_value
        list prides
        list nomads
        np.uint8_t[:] genders
        list history

    def __init__(self, objective_function, int dim, bounds, int population_size=50, int max_iter=100,
                 double nomad_ratio=0.2, int pride_size=5, double female_ratio=0.8, double roaming_ratio=0.2,
                 double mating_ratio=0.2, double mutation_prob=0.1, double immigration_ratio=0.1):
        """
        Initialize the Lion Optimization Algorithm (LOA).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: List of tuples [(low, high), ...] for each dimension.
        - population_size: Total number of lions (solutions).
        - max_iter: Maximum number of iterations.
        - nomad_ratio: Percentage of nomad lions (%N).
        - pride_size: Number of prides (P).
        - female_ratio: Percentage of females in prides (%S).
        - roaming_ratio: Percentage of territory for male roaming (%R).
        - mating_ratio: Percentage of females that mate (%Ma).
        - mutation_prob: Mutation probability for offspring (%Mu).
        - immigration_ratio: Ratio for female immigration between prides.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        if self.bounds.shape[0] != dim or self.bounds.shape[1] != 2:
            raise ValueError(f"Bounds must be a list of {dim} tuples, each containing (lower, upper) limits.")
        self.population_size = population_size
        self.max_iter = max_iter
        self.nomad_ratio = nomad_ratio
        self.pride_size = pride_size
        self.female_ratio = female_ratio
        self.roaming_ratio = roaming_ratio
        self.mating_ratio = mating_ratio
        self.mutation_prob = mutation_prob
        self.immigration_ratio = immigration_ratio
        self.global_best_value = np.inf
        self.prides = []
        self.nomads = []
        self.history = []

    cdef void initialize_population(self):
        """Generate initial lion population and organize into prides and nomads."""
        cdef:
            int i, num_nomads, num_females, num_nomad_females
            int idx
            double[:, :] lions
            double[:] fitness
            np.uint8_t[:] genders
            np.int32_t[:] indices
            np.int32_t[:] resident_indices
            np.int32_t[:] female_indices
            np.int32_t[:] nomad_female_indices
            list pride_indices

        self.lions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                       (self.population_size, self.dim))
        self.best_positions = self.lions.copy()
        self.best_fitness = np.empty(self.population_size, dtype=DTYPE)
        self.genders = np.zeros(self.population_size, dtype=np.uint8)
        self.global_best_solution = np.empty(self.dim, dtype=DTYPE)

        for i in range(self.population_size):
            self.best_fitness[i] = self.objective_function(np.asarray(self.lions[i]))

        num_nomads = <int>(self.nomad_ratio * self.population_size)
        indices = np.arange(self.population_size, dtype=np.int32)
        np.random.shuffle(indices)
        self.nomads = np.asarray(indices[:num_nomads]).tolist()
        resident_indices = indices[num_nomads:]

        self.prides = [np.asarray(resident_indices[i:i + self.pride_size]).tolist()
                       for i in range(0, len(resident_indices), self.pride_size)]

        for pride in self.prides:
            num_females = <int>(self.female_ratio * len(pride))
            female_indices = np.random.choice(pride, num_females, replace=False).astype(np.int32)
            for idx in female_indices:
                self.genders[idx] = 1
        num_nomad_females = <int>((1 - self.female_ratio) * len(self.nomads))
        nomad_female_indices = np.random.choice(self.nomads, num_nomad_females, replace=False).astype(np.int32)
        for idx in nomad_female_indices:
            self.genders[idx] = 1

    cdef void hunting(self, list pride_indices):
        """Simulate cooperative hunting in a pride."""
        cdef:
            int i, num_hunters, hunter_idx, size
            int idx
            list females, hunters, groups, center_group, wing_groups
            double[:] prey, new_pos, current_pos
            double new_fitness, old_fitness, pi
            double[:] hunter_fitness
            np.int32_t[:] sorted_hunters
            double group_sum, max_sum
            int max_group_idx

        females = [i for i in pride_indices if self.genders[i]]
        if not females:
            return
        num_hunters = max(1, len(females) // 2)
        hunters = np.random.choice(females, num_hunters, replace=False).astype(np.int32).tolist()

        hunter_fitness = np.array([self.best_fitness[i] for i in hunters], dtype=DTYPE)
        groups = []
        sorted_hunters = np.array(hunters, dtype=np.int32)[np.argsort(hunter_fitness)]
        size = len(hunters) // 3
        groups.append(np.asarray(sorted_hunters[:size + (len(hunters) % 3)]).tolist())
        groups.append(np.asarray(sorted_hunters[size + (len(hunters) % 3):2 * size + (len(hunters) % 3)]).tolist())
        groups.append(np.asarray(sorted_hunters[2 * size + (len(hunters) % 3):]).tolist())

        max_sum = -np.inf
        max_group_idx = 0
        for i in range(len(groups)):
            group = groups[i]
            group_sum = sum([self.best_fitness[j] for j in group])
            if group_sum > max_sum:
                max_sum = group_sum
                max_group_idx = i
        center_group = groups[max_group_idx]
        wing_groups = [g for g in groups if g != center_group]

        prey = np.mean([self.best_positions[i] for i in hunters], axis=0)

        new_pos = np.empty(self.dim, dtype=DTYPE)
        for hunter_idx in hunters:
            current_pos = self.lions[hunter_idx]
            if hunter_idx in center_group:
                for i in range(self.dim):
                    if current_pos[i] < prey[i]:
                        new_pos[i] = current_pos[i] + rand_uniform() * (prey[i] - current_pos[i])
                    else:
                        new_pos[i] = prey[i] + rand_uniform() * (current_pos[i] - prey[i])
            else:
                for i in range(self.dim):
                    new_pos[i] = self.bounds[i, 0] + self.bounds[i, 1] - current_pos[i]

            new_fitness = self.objective_function(np.asarray(new_pos))
            old_fitness = self.objective_function(np.asarray(current_pos))
            if new_fitness < old_fitness:
                for i in range(self.dim):
                    self.lions[hunter_idx, i] = new_pos[i]
                if new_fitness < self.best_fitness[hunter_idx]:
                    for i in range(self.dim):
                        self.best_positions[hunter_idx, i] = new_pos[i]
                    self.best_fitness[hunter_idx] = new_fitness
                pi = (old_fitness - new_fitness) / old_fitness if old_fitness != 0 else 1
                for i in range(self.dim):
                    prey[i] += rand_uniform() * pi * (prey[i] - new_pos[i])

    cdef void move_to_safe_place(self, list pride_indices):
        """Move non-hunting females toward pride territory."""
        cdef:
            int i, idx, selected_idx, tournament_size, k
            list females, hunters, non_hunters, candidates
            double[:] current_pos, selected_pos, new_pos, r1, r2
            double d, theta, norm_r1, norm_r2
            double new_fitness
            np.int32_t[:] success
            np.int32_t[:] candidates_array

        females = [i for i in pride_indices if self.genders[i]]
        if not females:
            return

        hunters = np.random.choice(females, max(1, len(females) // 2), replace=False).astype(np.int32).tolist()
        non_hunters = [i for i in females if i not in hunters]
        if not non_hunters:
            return

        success = np.array([1 if self.best_fitness[i] < self.objective_function(np.asarray(self.lions[i]))
                           else 0 for i in pride_indices], dtype=np.int32)
        k = np.sum(success)
        tournament_size = max(2, <int>ceil(k / 2.0))

        new_pos = np.empty(self.dim, dtype=DTYPE)
        r1 = np.empty(self.dim, dtype=DTYPE)
        r2 = np.empty(self.dim, dtype=DTYPE)
        for idx in non_hunters:
            candidates_array = np.random.choice(pride_indices, tournament_size, replace=False).astype(np.int32)
            candidates = np.asarray(candidates_array).tolist()
            selected_idx = candidates[0]
            for i in range(1, tournament_size):
                if self.best_fitness[candidates[i]] < self.best_fitness[selected_idx]:
                    selected_idx = candidates[i]
            selected_pos = self.best_positions[selected_idx]

            current_pos = self.lions[idx]
            d = 0.0
            for i in range(self.dim):
                d += (selected_pos[i] - current_pos[i]) ** 2
            d = sqrt(d)

            norm_r1 = 0.0
            for i in range(self.dim):
                r1[i] = selected_pos[i] - current_pos[i]
                norm_r1 += r1[i] ** 2
            norm_r1 = sqrt(norm_r1) if norm_r1 != 0 else 1e-10
            for i in range(self.dim):
                r1[i] /= norm_r1

            for i in range(self.dim):
                r2[i] = rand_uniform() * 2 - 1
            dot = 0.0
            for i in range(self.dim):
                dot += r2[i] * r1[i]
            for i in range(self.dim):
                r2[i] -= dot * r1[i]
            norm_r2 = 0.0
            for i in range(self.dim):
                norm_r2 += r2[i] ** 2
            norm_r2 = sqrt(norm_r2) if norm_r2 != 0 else 1e-10
            for i in range(self.dim):
                r2[i] /= norm_r2

            theta = (rand_uniform() - 0.5) * np.pi
            for i in range(self.dim):
                new_pos[i] = (current_pos[i] + 2 * d * rand_uniform() * r1[i] +
                              (rand_uniform() * 2 - 1) * tan(theta) * d * r2[i])
                new_pos[i] = min(max(new_pos[i], self.bounds[i, 0]), self.bounds[i, 1])

            new_fitness = self.objective_function(np.asarray(new_pos))
            if new_fitness < self.best_fitness[idx]:
                for i in range(self.dim):
                    self.lions[idx, i] = new_pos[i]
                    self.best_positions[idx, i] = new_pos[i]
                self.best_fitness[idx] = new_fitness

    cdef void roaming(self, list pride_indices):
        """Simulate male lions roaming in pride territory."""
        cdef:
            int i, idx, j, num_visits
            list males, territory
            double[:] target, direction, new_pos
            double d, theta, x
            double new_fitness
            np.int32_t[:] visit_indices

        males = [i for i in pride_indices if not self.genders[i]]
        new_pos = np.empty(self.dim, dtype=DTYPE)
        direction = np.empty(self.dim, dtype=DTYPE)
        for idx in males:
            territory = [self.best_positions[i] for i in pride_indices]
            num_visits = <int>(self.roaming_ratio * len(territory))
            visit_indices = np.random.choice(len(territory), num_visits, replace=False).astype(np.int32)

            for j in visit_indices:
                target = territory[j]
                d = 0.0
                for i in range(self.dim):
                    d += (target[i] - self.lions[idx, i]) ** 2
                d = sqrt(d)

                norm = 0.0
                for i in range(self.dim):
                    direction[i] = target[i] - self.lions[idx, i]
                    norm += direction[i] ** 2
                norm = sqrt(norm) if norm != 0 else 1e-10
                for i in range(self.dim):
                    direction[i] /= norm

                theta = (rand_uniform() - 0.5) * np.pi / 3
                x = rand_uniform() * 2 * d

                if self.dim >= 2:
                    c = cos(theta)
                    s = sin(theta)
                    temp0 = direction[0] * c - direction[1] * s
                    temp1 = direction[0] * s + direction[1] * c
                    direction[0] = temp0
                    direction[1] = temp1

                for i in range(self.dim):
                    new_pos[i] = self.lions[idx, i] + x * direction[i]
                    new_pos[i] = min(max(new_pos[i], self.bounds[i, 0]), self.bounds[i, 1])

                new_fitness = self.objective_function(np.asarray(new_pos))
                if new_fitness < self.best_fitness[idx]:
                    for i in range(self.dim):
                        self.lions[idx, i] = new_pos[i]
                        self.best_positions[idx, i] = new_pos[i]
                    self.best_fitness[idx] = new_fitness

    cdef void nomad_movement(self):
        """Simulate random movement of nomad lions."""
        cdef:
            int i, j, idx
            double pr, new_fitness
            double[:] new_pos
            double min_fitness, max_fitness

        new_pos = np.empty(self.dim, dtype=DTYPE)
        min_fitness = np.min(self.best_fitness)
        max_fitness = np.max(self.best_fitness)
        for idx in self.nomads:
            pr = (self.objective_function(np.asarray(self.lions[idx])) - min_fitness) / \
                 (max_fitness - min_fitness + 1e-10)
            for j in range(self.dim):
                new_pos[j] = self.lions[idx, j]
                if rand_uniform() > pr:
                    new_pos[j] = self.bounds[j, 0] + rand_uniform() * (self.bounds[j, 1] - self.bounds[j, 0])
            new_fitness = self.objective_function(np.asarray(new_pos))
            if new_fitness < self.best_fitness[idx]:
                for j in range(self.dim):
                    self.lions[idx, j] = new_pos[j]
                    self.best_positions[idx, j] = new_pos[j]
                self.best_fitness[idx] = new_fitness

    cdef void mating(self, list pride_indices):
        """Simulate mating to produce offspring."""
        cdef:
            int i, female_idx, num_mating, num_mates
            int j, m
            list females, males, mating_females, selected_males
            double beta, sum_s
            double[:] offspring1, offspring2, s
            double new_fitness1, new_fitness2

        females = [i for i in pride_indices if self.genders[i]]
        num_mating = <int>(self.mating_ratio * len(females))
        mating_females = np.random.choice(females, num_mating, replace=False).astype(np.int32).tolist()
        males = [i for i in pride_indices if not self.genders[i]]

        offspring1 = np.empty(self.dim, dtype=DTYPE)
        offspring2 = np.empty(self.dim, dtype=DTYPE)
        print(f"Mating: len(lions)={len(self.lions)}, population_size={self.population_size}, num_mating={num_mating}")
        for female_idx in mating_females:
            if not males:
                continue
            num_mates = np.random.randint(1, len(males) + 1)
            selected_males = np.random.choice(males, num_mates, replace=False).astype(np.int32).tolist()
            beta = np.random.normal(0.5, 0.1)
            s = np.zeros(len(males), dtype=DTYPE)
            for m in selected_males:
                s[males.index(m)] = 1
            sum_s = np.sum(s) + 1e-10

            for i in range(self.dim):
                male_sum = 0.0
                for j in range(len(males)):
                    m = males[j]
                    male_sum += self.lions[m, i] * s[j]
                offspring1[i] = beta * self.lions[female_idx, i] + (1 - beta) * male_sum / sum_s
                offspring2[i] = (1 - beta) * self.lions[female_idx, i] + beta * male_sum / sum_s

            for j in range(self.dim):
                if rand_uniform() < self.mutation_prob:
                    offspring1[j] = self.bounds[j, 0] + rand_uniform() * (self.bounds[j, 1] - self.bounds[j, 0])
                    offspring2[j] = self.bounds[j, 0] + rand_uniform() * (self.bounds[j, 1] - self.bounds[j, 0])

            for i in range(self.dim):
                offspring1[i] = min(max(offspring1[i], self.bounds[i, 0]), self.bounds[i, 1])
                offspring2[i] = min(max(offspring2[i], self.bounds[i, 0]), self.bounds[i, 1])

            if len(self.lions) <= self.population_size:
                print(f"Adding offspring: current_len={len(self.lions)}, new_indices=[{len(self.lions)}, {len(self.lions)+1}]")
                self.lions = np.vstack([self.lions, offspring1, offspring2])
                self.best_positions = np.vstack([self.best_positions, offspring1, offspring2])
                new_fitness1 = self.objective_function(np.asarray(offspring1))
                new_fitness2 = self.objective_function(np.asarray(offspring2))
                self.best_fitness = np.append(self.best_fitness, [new_fitness1, new_fitness2])
                self.genders = np.append(self.genders, [True, False])
                pride_indices.extend([len(self.lions) - 2, len(self.lions) - 1])
                print(f"After adding: len(lions)={len(self.lions)}, pride_indices={pride_indices}")

    cdef void defense(self, list pride_indices):
        """Simulate defense against mature males and nomad invasions."""
        cdef:
            int i, weakest_male, nomad_idx, resident_idx
            list males, nomad_males, male_fitness

        males = [i for i in pride_indices if not self.genders[i]]
        if len(males) <= 1:
            return

        male_fitness = [(i, self.best_fitness[i]) for i in pride_indices if not self.genders[i]]
        male_fitness.sort(key=lambda x: x[1])
        weakest_male = male_fitness[-1][0]
        if weakest_male in pride_indices:
            self.nomads.append(weakest_male)
            pride_indices.remove(weakest_male)

        nomad_males = [i for i in self.nomads if not self.genders[i]]
        for nomad_idx in nomad_males[:]:
            if rand_uniform() < 0.5:
                for resident_idx in males[:]:
                    if resident_idx in pride_indices and nomad_idx in self.nomads:
                        if self.best_fitness[nomad_idx] < self.best_fitness[resident_idx]:
                            self.nomads.remove(nomad_idx)
                            pride_indices.append(nomad_idx)
                            self.nomads.append(resident_idx)
                            if resident_idx in pride_indices:
                                pride_indices.remove(resident_idx)
                            break

    cdef void immigration(self):
        """Simulate female immigration between prides or to nomads."""
        cdef:
            int i, idx, num_immigrants, other_pride_idx
            list females, immigrants, nomad_females

        for i in range(len(self.prides)):
            pride = self.prides[i]
            females = [idx for idx in pride if self.genders[idx]]
            num_immigrants = <int>(self.immigration_ratio * len(females))
            immigrants = np.random.choice(females, num_immigrants, replace=False).astype(np.int32).tolist()
            for idx in immigrants:
                if rand_uniform() < 0.5:
                    pride.remove(idx)
                    if rand_uniform() < 0.5 and len(self.prides) > 1:
                        other_pride_idx = np.random.randint(0, len(self.prides))
                        while other_pride_idx == i:
                            other_pride_idx = np.random.randint(0, len(self.prides))
                        self.prides[other_pride_idx].append(idx)
                    else:
                        self.nomads.append(idx)
        nomad_females = [idx for idx in self.nomads if self.genders[idx]]
        for idx in nomad_females[:]:
            if rand_uniform() < 0.1 and self.prides:
                self.nomads.remove(idx)
                random_pride_idx = np.random.randint(0, len(self.prides))
                self.prides[random_pride_idx].append(idx)

    cdef void population_control(self):
        """Eliminate weakest lions to maintain population size."""
        cdef:
            int i, j, k, num_excess, new_size
            np.int32_t[:] worst_indices
            double[:, :] new_lions
            double[:, :] new_best_positions
            double[:] new_best_fitness
            np.uint8_t[:] new_genders
            list new_nomads, new_prides, new_pride

        print(f"Population control: len(lions)={len(self.lions)}, population_size={self.population_size}")
        if len(self.lions) > self.population_size:
            num_excess = len(self.lions) - self.population_size
            new_size = self.population_size
            worst_indices = np.argsort(self.best_fitness).astype(np.int32)[len(self.best_fitness)-num_excess:]
            print(f"Removing {num_excess} lions, worst_indices={np.asarray(worst_indices).tolist()}")
            if np.any(np.asarray(worst_indices) >= len(self.lions)) or np.any(np.asarray(worst_indices) < 0):
                raise ValueError(f"Invalid worst_indices: {np.asarray(worst_indices).tolist()}")

            new_lions = np.empty((new_size, self.dim), dtype=DTYPE)
            new_best_positions = np.empty((new_size, self.dim), dtype=DTYPE)
            new_best_fitness = np.empty(new_size, dtype=DTYPE)
            new_genders = np.empty(new_size, dtype=np.uint8)

            k = 0
            for i in range(len(self.lions)):
                keep = 1
                for j in range(num_excess):
                    if i == worst_indices[j]:
                        keep = 0
                        break
                if keep:
                    for j in range(self.dim):
                        new_lions[k, j] = self.lions[i, j]
                        new_best_positions[k, j] = self.best_positions[i, j]
                    new_best_fitness[k] = self.best_fitness[i]
                    new_genders[k] = self.genders[i]
                    k += 1

            self.lions = new_lions
            self.best_positions = new_best_positions
            self.best_fitness = new_best_fitness
            self.genders = new_genders

            new_nomads = [idx for idx in self.nomads if idx < new_size]
            self.nomads = new_nomads
            new_prides = []
            for pride in self.prides:
                new_pride = [idx for idx in pride if idx < new_size]
                if new_pride:
                    new_prides.append(new_pride)
            self.prides = new_prides
            print(f"After population control: len(lions)={len(self.lions)}, nomads={self.nomads}, prides={self.prides}")

    def optimize(self):
        """Run the Lion Optimization Algorithm."""
        cdef:
            int generation, min_idx
            double[:] global_best_solution
            double global_best_value

        self.initialize_population()
        for generation in range(self.max_iter):
            min_idx = np.argmin(self.best_fitness)
            if self.best_fitness[min_idx] < self.global_best_value:
                for i in range(self.dim):
                    self.global_best_solution[i] = self.best_positions[min_idx, i]
                self.global_best_value = self.best_fitness[min_idx]

            for pride in self.prides[:]:
                self.hunting(pride)
                self.move_to_safe_place(pride)
                self.roaming(pride)
                self.mating(pride)
                self.defense(pride)

            self.nomad_movement()
            self.immigration()
            self.population_control()

            self.history.append((generation, np.asarray(self.global_best_solution).copy(), self.global_best_value))
            print(f"Iteration {generation + 1}: Best Value = {self.global_best_value}")

        return np.asarray(self.global_best_solution), self.global_best_value, self.history
