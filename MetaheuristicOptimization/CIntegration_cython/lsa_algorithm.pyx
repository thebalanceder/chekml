import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, abs, sqrt
from libc.stdlib cimport rand, RAND_MAX

# Define numpy types
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double evaluate_channel(np.ndarray[DTYPE_t, ndim=1] channel, object objective_function):
    """Evaluate a single channel using the objective function."""
    return objective_function(channel)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void initialize_channels(
    np.ndarray[DTYPE_t, ndim=2] channels,
    np.ndarray[DTYPE_t, ndim=2] bounds,
    np.ndarray[DTYPE_t, ndim=2] directions,
    int population_size,
    int dim
):
    """Generate initial lightning channels and directions randomly."""
    cdef int i, d
    cdef double lb, ub
    for i in range(population_size):
        for d in range(dim):
            lb = bounds[d, 0]
            ub = bounds[d, 1]
            channels[i, d] = lb + (ub - lb) * (<double>rand() / RAND_MAX)
    for d in range(dim):
        directions[0, d] = -1.0 + 2.0 * (<double>rand() / RAND_MAX)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_t, ndim=1] evaluate_channels(
    np.ndarray[DTYPE_t, ndim=2] channels,
    object objective_function,
    int population_size
):
    """Compute fitness values for all channels."""
    cdef np.ndarray[DTYPE_t, ndim=1] fitness = np.empty(population_size, dtype=DTYPE)
    cdef int i
    for i in range(population_size):
        fitness[i] = evaluate_channel(channels[i], objective_function)
    return fitness

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_channel_elimination(
    np.ndarray[DTYPE_t, ndim=2] channels,
    np.ndarray[DTYPE_t, ndim=1] fitness,
    int* channel_time,
    int max_channel_time,
    int population_size
):
    """Eliminate the worst channel after max_channel_time iterations."""
    channel_time[0] += 1
    if channel_time[0] >= max_channel_time:
        sorted_indices = np.argsort(fitness)
        worst_idx = sorted_indices[population_size - 1]
        best_idx = sorted_indices[0]
        channels[worst_idx] = channels[best_idx].copy()
        fitness[worst_idx] = fitness[best_idx]
        channel_time[0] = 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_directions(
    np.ndarray[DTYPE_t, ndim=2] channels,
    np.ndarray[DTYPE_t, ndim=1] fitness,
    np.ndarray[DTYPE_t, ndim=2] directions,
    np.ndarray[DTYPE_t, ndim=2] bounds,
    int min_idx,
    object objective_function,
    int dim
):
    """Update the direction of lightning movement."""
    cdef int d
    cdef double test_fitness, best_fitness = fitness[min_idx]
    cdef np.ndarray[DTYPE_t, ndim=1] test_channel = channels[min_idx].copy()
    for d in range(dim):
        test_channel[d] = channels[min_idx, d] + directions[0, d] * 0.005 * (bounds[d, 1] - bounds[d, 0])
        test_fitness = evaluate_channel(test_channel, objective_function)
        if test_fitness < best_fitness:
            directions[0, d] = directions[0, d]
        else:
            directions[0, d] = -directions[0, d]
        test_channel[d] = channels[min_idx, d]  # Reset for next iteration

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_positions(
    np.ndarray[DTYPE_t, ndim=2] channels,
    np.ndarray[DTYPE_t, ndim=1] fitness,
    np.ndarray[DTYPE_t, ndim=2] directions,
    np.ndarray[DTYPE_t, ndim=2] bounds,
    int best_idx,
    object objective_function,
    int population_size,
    int dim,
    int t,
    int max_iter,
    double energy_factor
):
    """Update channel positions based on energy and distance."""
    cdef double energy = energy_factor - 2.0 * exp(-5.0 * (max_iter - t) / max_iter)
    cdef int i, d
    cdef double dist_d, temp_fitness, fock_fitness
    cdef np.ndarray[DTYPE_t, ndim=1] temp_channel = np.empty(dim, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] fock_channel = np.empty(dim, dtype=DTYPE)
    cdef double r, lb, ub

    for i in range(population_size):
        for d in range(dim):
            dist_d = channels[i, d] - channels[best_idx, d]
            if abs(dist_d) < 1e-10:  # Channels are equal
                temp_channel[d] = channels[i, d] + directions[0, d] * abs(np.random.normal(0, energy))
            else:
                r = np.random.exponential(abs(dist_d))
                if dist_d < 0:
                    temp_channel[d] = channels[i, d] + r
                else:
                    temp_channel[d] = channels[i, d] - r

            # Boundary check
            lb = bounds[d, 0]
            ub = bounds[d, 1]
            if temp_channel[d] > ub or temp_channel[d] < lb:
                temp_channel[d] = lb + (ub - lb) * (<double>rand() / RAND_MAX)

        temp_fitness = evaluate_channel(temp_channel, objective_function)
        if temp_fitness < fitness[i]:
            channels[i] = temp_channel.copy()
            fitness[i] = temp_fitness

            # Focking procedure
            if (<double>rand() / RAND_MAX) < 0.01:
                for d in range(dim):
                    fock_channel[d] = bounds[d, 0] + bounds[d, 1] - temp_channel[d]
                fock_fitness = evaluate_channel(fock_channel, objective_function)
                if fock_fitness < fitness[i]:
                    channels[i] = fock_channel.copy()
                    fitness[i] = fock_fitness

cdef class LightningSearchAlgorithm:
    cdef object objective_function
    cdef int dim, population_size, max_iter, max_channel_time
    cdef double energy_factor
    cdef np.ndarray bounds, channels, directions
    cdef np.ndarray best_solution
    cdef double best_value
    cdef list history
    cdef int channel_time

    def __init__(self, objective_function, dim, bounds, population_size=50, max_iter=500, 
                 max_channel_time=10, energy_factor=2.05):
        """
        Initialize the Lightning Search Algorithm (LSA).

        Parameters:
        - objective_function: Function to optimize.
        - dim: Number of dimensions (variables).
        - bounds: Array of (lower, upper) bounds for each dimension.
        - population_size: Number of lightning channels (solutions).
        - max_iter: Maximum number of iterations.
        - max_channel_time: Maximum time before channel elimination.
        - energy_factor: Initial energy factor for movement.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.bounds = np.array(bounds, dtype=DTYPE)
        self.population_size = population_size
        self.max_iter = max_iter
        self.max_channel_time = max_channel_time
        self.energy_factor = energy_factor
        self.best_value = float("inf")
        self.history = []
        self.channel_time = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def optimize(self):
        """Run the Lightning Search Algorithm."""
        import matplotlib.pyplot as plt
        import time
        cdef double start_time = time.time()
        cdef int t, min_idx, max_idx
        cdef np.ndarray[DTYPE_t, ndim=2] channels = np.empty((self.population_size, self.dim), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] directions = np.empty((1, self.dim), dtype=DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] fitness

        # Initialize channels and directions
        initialize_channels(channels, self.bounds, directions, self.population_size, self.dim)
        self.channels = channels
        self.directions = directions

        for t in range(self.max_iter):
            # Evaluate fitness
            fitness = evaluate_channels(self.channels, self.objective_function, self.population_size)
            min_idx = np.argmin(fitness)
            max_idx = np.argmax(fitness)

            # Update best solution
            if fitness[min_idx] < self.best_value:
                self.best_solution = self.channels[min_idx].copy()
                self.best_value = fitness[min_idx]

            # Update channel elimination
            update_channel_elimination(self.channels, fitness, &self.channel_time, 
                                     self.max_channel_time, self.population_size)

            # Update directions
            update_directions(self.channels, fitness, self.directions, self.bounds, 
                            min_idx, self.objective_function, self.dim)

            # Update positions
            update_positions(self.channels, fitness, self.directions, self.bounds, min_idx,
                           self.objective_function, self.population_size, self.dim, t,
                           self.max_iter, self.energy_factor)

            # Stop if best and worst are equal
            if abs(fitness[min_idx] - fitness[max_idx]) < 1e-10:
                break

            # Record history
            self.history.append((t, self.channels[min_idx].copy(), fitness[min_idx]))

        cdef double elapsed_time = time.time() - start_time

        # Plot convergence
        iterations, _, values = zip(*self.history)
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, values, 'b-', linewidth=2)
        plt.xlabel('No of Iteration')
        plt.ylabel('Fitness Value')
        plt.title('Convergence of Lightning Search Algorithm')
        plt.grid(True)
        plt.savefig('lsa_convergence.png')

        print(f"Optimal value = {self.best_value}")
        return self.best_solution, self.best_value, self.history, elapsed_time
