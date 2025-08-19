// Constants
#define ARFO_BRANCHING_THRESHOLD 0.1f
#define ARFO_MAX_BRANCHING 5
#define ARFO_MIN_BRANCHING 1
#define ARFO_INITIAL_STD 1.0f
#define ARFO_FINAL_STD 0.01f
#define ARFO_LOCAL 0.1f

typedef struct {
    float fitness;
    int index;
} ARFOFitnessIndex;

// Random number generator
float rand_float(__global uint *seed) {
    *seed = (*seed * 1103515245U + 12345U) & 0x7fffffffU;
    return (float)(*seed) / (float)0x7fffffff;
}

// Calculate auxin concentration
void calculate_auxin_concentration(__global float *fitness, __global float *auxin, int pop_size) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    float f_min = fitness[0];
    float f_max = fitness[0];
    for (int i = 1; i < pop_size; i++) {
        if (fitness[i] < f_min) f_min = fitness[i];
        if (fitness[i] > f_max) f_max = fitness[i];
    }

    float sum_f = 0.0f;
    for (int i = 0; i < pop_size; i++) {
        auxin[i] = (fitness[i] - f_min) / (f_max - f_min + 1e-10f);
        sum_f += auxin[i];
    }
    for (int i = 0; i < pop_size; i++) {
        auxin[i] = sum_f > 0.0f ? (auxin[i] / sum_f * pop_size) : 0.0f;
    }
}

__kernel void initialize_population(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global float *best_position,
    __global float *best_fitness,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        population[gid * dim + j] = lower + rand_float(&seeds[gid]) * (upper - lower);
    }

    if (gid == 0) {
        *best_fitness = INFINITY;
        for (int j = 0; j < dim; j++) {
            best_position[j] = population[j];
        }
    }
}

__kernel void regrowth_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global float *auxin,
    __global float *auxin_sorted,
    __global int *topology,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    calculate_auxin_concentration(fitness, auxin, pop_size);

    // Copy auxin to auxin_sorted
    for (int i = 0; i < pop_size; i++) {
        auxin_sorted[i] = auxin[i];
    }

    // Sort auxin_sorted (bubble sort for simplicity)
    for (int i = 0; i < pop_size - 1; i++) {
        for (int j = 0; j < pop_size - i - 1; j++) {
            if (auxin_sorted[j] > auxin_sorted[j + 1]) {
                float temp = auxin_sorted[j];
                auxin_sorted[j] = auxin_sorted[j + 1];
                auxin_sorted[j + 1] = temp;
            }
        }
    }

    // Use median auxin for movement decision
    float median_aux = auxin_sorted[pop_size / 2];

    // Identify valid neighbors
    float valid_neighbors[4];
    int neighbor_count = 0;
    for (int k = 0; k < 4; k++) {
        if (topology[gid * 4 + k] >= 0) {
            valid_neighbors[neighbor_count++] = topology[gid * 4 + k];
        }
    }

    // Move towards best neighbor if auxin is high
    if (auxin[gid] > median_aux && neighbor_count > 0) {
        float min_f = fitness[(int)valid_neighbors[0]];
        int best_neighbor = (int)valid_neighbors[0];
        for (int i = 1; i < neighbor_count; i++) {
            int idx = (int)valid_neighbors[i];
            if (fitness[idx] < min_f) {
                min_f = fitness[idx];
                best_neighbor = idx;
            }
        }

        // Update position
        float rand_coeff = rand_float(&seeds[gid]);
        for (int j = 0; j < dim; j++) {
            float lower = bounds[j * 2];
            float upper = bounds[j * 2 + 1];
            population[gid * dim + j] += ARFO_LOCAL * rand_coeff * (population[best_neighbor * dim + j] - population[gid * dim + j]);
            population[gid * dim + j] = clamp(population[gid * dim + j], lower, upper);
        }
    }
}

__kernel void branching_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global float *auxin,
    __global float *new_roots,
    __global ARFOFitnessIndex *fitness_indices,
    __global uint *seeds,
    int dim,
    int pop_size,
    int iter,
    int max_iter,
    __global int *new_root_count
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    calculate_auxin_concentration(fitness, auxin, pop_size);

    int local_new_root_count = 0;
    if (auxin[gid] > ARFO_BRANCHING_THRESHOLD) {
        float R1 = rand_float(&seeds[gid]);
        int num_new_roots = (int)(R1 * auxin[gid] * (ARFO_MAX_BRANCHING - ARFO_MIN_BRANCHING)) + ARFO_MIN_BRANCHING;
        float std = ((float)(max_iter - iter) / max_iter) * (ARFO_INITIAL_STD - ARFO_FINAL_STD) + ARFO_FINAL_STD;

        for (int k = 0; k < num_new_roots && k < ARFO_MAX_BRANCHING; k++) {
            int idx = atomic_add(new_root_count, 1);
            if (idx < pop_size * ARFO_MAX_BRANCHING) {
                for (int j = 0; j < dim; j++) {
                    new_roots[idx * dim + j] = population[gid * dim + j] + std * (rand_float(&seeds[gid]) * 2.0f - 1.0f);
                    new_roots[idx * dim + j] = clamp(new_roots[idx * dim + j], bounds[j * 2], bounds[j * 2 + 1]);
                }
                local_new_root_count++;
            }
        }
    }

    // Update fitness_indices
    fitness_indices[gid].fitness = fitness[gid];
    fitness_indices[gid].index = gid;
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Simplified bubble sort for fitness_indices
    for (int i = 0; i < pop_size - 1; i++) {
        for (int j = 0; j < pop_size - i - 1; j++) {
            if (fitness_indices[j].fitness > fitness_indices[j + 1].fitness) {
                ARFOFitnessIndex temp = fitness_indices[j];
                fitness_indices[j] = fitness_indices[j + 1];
                fitness_indices[j + 1] = temp;
            }
        }
    }
}

__kernel void lateral_growth_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global float *auxin,
    __global float *auxin_sorted,
    __global float *new_roots,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    calculate_auxin_concentration(fitness, auxin, pop_size);

    float R1 = rand_float(&seeds[gid]);
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        population[gid * dim + j] += R1 * (upper - lower) * 0.01f;
        population[gid * dim + j] = clamp(population[gid * dim + j], lower, upper);
    }
}

__kernel void elimination_phase(
    __global float *population,
    __global float *fitness,
    __global float *auxin,
    __global float *auxin_sorted,
    int dim,
    int pop_size,
    __global int *new_pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    calculate_auxin_concentration(fitness, auxin, pop_size);

    // Copy auxin to auxin_sorted
    for (int i = 0; i < pop_size; i++) {
        auxin_sorted[i] = auxin[i];
    }

    // Sort auxin_sorted
    for (int i = 0; i < pop_size - 1; i++) {
        for (int j = 0; j < pop_size - i - 1; j++) {
            if (auxin_sorted[j] > auxin_sorted[j + 1]) {
                float temp = auxin_sorted[j];
                auxin_sorted[j] = auxin_sorted[j + 1];
                auxin_sorted[j + 1] = temp;
            }
        }
    }

    float threshold = auxin_sorted[pop_size / 2];
    if (auxin[gid] < threshold) {
        // Mark for elimination by setting fitness to a high value
        fitness[gid] = INFINITY;
    } else {
        atomic_inc(new_pop_size);
    }
}

__kernel void replenish_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    __global int *pop_size,
    int original_pop_size
) {
    int gid = get_global_id(0);
    if (gid >= original_pop_size) return;

    int current_pop_size = *pop_size;
    if (gid < current_pop_size && fitness[gid] == INFINITY) {
        for (int j = 0; j < dim; j++) {
            float lower = bounds[j * 2];
            float upper = bounds[j * 2 + 1];
            population[gid * dim + j] = lower + rand_float(&seeds[gid]) * (upper - lower);
        }
        fitness[gid] = 0.0f; // Reset fitness
    } else if (gid >= current_pop_size && gid < original_pop_size) {
        for (int j = 0; j < dim; j++) {
            float lower = bounds[j * 2];
            float upper = bounds[j * 2 + 1];
            population[gid * dim + j] = lower + rand_float(&seeds[gid]) * (upper - lower);
        }
        fitness[gid] = 0.0f;
        atomic_inc(pop_size);
    }
}
