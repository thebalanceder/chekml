#define LOA_NOMAD_RATIO 0.2f
#define LOA_PRIDE_SIZE 5
#define LOA_FEMALE_RATIO 0.8f
#define LOA_ROAMING_RATIO 0.2f
#define LOA_MATING_RATIO 0.2f
#define LOA_MUTATION_PROB 0.1f
#define LOA_IMMIGRATION_RATIO 0.1f
#define M_PI 3.14159265358979323846f

// Random number generator (XOR-shift)
inline float rand_float(uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}

// Initialize population
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

    // Initialize position
    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
    }

    // Placeholder fitness (evaluated on CPU)
    fitness[gid] = INFINITY;
}

// Hunting phase
__kernel void hunting_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    __global int *prides,
    __global int *pride_sizes,
    __global uchar *genders,
    __global float *temp_buffer,
    __global int *females,
    __global int *hunters,
    int pride_idx,
    int dim,
    int pop_size
) {
    int lid = get_global_id(0);
    if (lid >= pride_sizes[pride_idx]) return;

    int lion_idx = prides[pride_idx * LOA_PRIDE_SIZE + lid];

    // Collect females
    int num_females = 0;
    for (int i = 0; i < pride_sizes[pride_idx]; i++) {
        int idx = prides[pride_idx * LOA_PRIDE_SIZE + i];
        if (genders[idx]) {
            females[num_females++] = idx;
        }
    }
    if (num_females == 0) return;

    // Select hunters
    int num_hunters = num_females / 2 > 0 ? num_females / 2 : 1;
    for (int i = 0; i < num_hunters; i++) {
        hunters[i] = females[(int)(rand_float(&seeds[lion_idx]) * num_females)];
    }

    // Compute prey position
    for (int j = 0; j < dim; j++) temp_buffer[j] = 0.0f;
    for (int i = 0; i < num_hunters; i++) {
        for (int j = 0; j < dim; j++) {
            temp_buffer[j] += population[hunters[i] * dim + j];
        }
    }
    for (int j = 0; j < dim; j++) {
        temp_buffer[j] /= num_hunters;
    }

    // Update hunter positions
    for (int i = 0; i < num_hunters; i++) {
        if (lion_idx != hunters[i]) continue;
        float old_fitness = fitness[lion_idx];
        __global float *pos = population + lion_idx * dim;

        for (int j = 0; j < dim; j++) {
            float p = temp_buffer[j];
            float move = rand_float(&seeds[lion_idx]) * (p - pos[j]) * (pos[j] < p ? 1.0f : -1.0f);
            pos[j] += move;
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }

        // Placeholder fitness (evaluated on CPU)
        fitness[lion_idx] = INFINITY;
    }
}

// Move to safe place phase
__kernel void move_to_safe_place_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    __global int *prides,
    __global int *pride_sizes,
    __global uchar *genders,
    __global int *females,
    __global int *hunters,
    __global int *non_hunters,
    int pride_idx,
    int dim,
    int pop_size
) {
    int lid = get_global_id(0);
    if (lid >= pride_sizes[pride_idx]) return;

    int lion_idx = prides[pride_idx * LOA_PRIDE_SIZE + lid];

    // Collect females
    int num_females = 0;
    for (int i = 0; i < pride_sizes[pride_idx]; i++) {
        int idx = prides[pride_idx * LOA_PRIDE_SIZE + i];
        if (genders[idx]) {
            females[num_females++] = idx;
        }
    }
    if (num_females == 0) return;

    // Select hunters
    int num_hunters = num_females / 2 > 0 ? num_females / 2 : 1;
    for (int i = 0; i < num_hunters; i++) {
        hunters[i] = females[(int)(rand_float(&seeds[lion_idx]) * num_females)];
    }

    // Collect non-hunters
    int num_non_hunters = 0;
    for (int i = 0; i < num_females; i++) {
        int is_hunter = 0;
        for (int j = 0; j < num_hunters; j++) {
            if (females[i] == hunters[j]) {
                is_hunter = 1;
                break;
            }
        }
        if (!is_hunter) {
            non_hunters[num_non_hunters++] = females[i];
        }
    }

    // Update non-hunter positions
    for (int i = 0; i < num_non_hunters; i++) {
        if (lion_idx != non_hunters[i]) continue;
        __global float *pos = population + lion_idx * dim;
        int selected_idx = prides[pride_idx * LOA_PRIDE_SIZE + (int)(rand_float(&seeds[lion_idx]) * pride_sizes[pride_idx])];
        float d = 0.0f;
        for (int j = 0; j < dim; j++) {
            float diff = population[selected_idx * dim + j] - pos[j];
            d += diff * diff;
        }
        d = sqrt(d);

        float norm_r1 = 0.0f;
        float r1[32]; // Assuming dim <= 32
        for (int j = 0; j < dim; j++) {
            r1[j] = population[selected_idx * dim + j] - pos[j];
            norm_r1 += r1[j] * r1[j];
        }
        norm_r1 = norm_r1 > 0.0f ? sqrt(norm_r1) : 1e-10f;
        for (int j = 0; j < dim; j++) {
            r1[j] /= norm_r1;
        }

        float theta = (rand_float(&seeds[lion_idx]) - 0.5f) * M_PI;
        float tan_theta = tan(theta);
        for (int j = 0; j < dim; j++) {
            pos[j] += 2.0f * d * rand_float(&seeds[lion_idx]) * r1[j] +
                      rand_float(&seeds[lion_idx]) * tan_theta * d * r1[j];
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }

        // Placeholder fitness (evaluated on CPU)
        fitness[lion_idx] = INFINITY;
    }
}

// Roaming phase
__kernel void roaming_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    __global int *prides,
    __global int *pride_sizes,
    __global uchar *genders,
    __global int *males,
    int pride_idx,
    int dim,
    int pop_size
) {
    int lid = get_global_id(0);
    if (lid >= pride_sizes[pride_idx]) return;

    int lion_idx = prides[pride_idx * LOA_PRIDE_SIZE + lid];

    // Collect males
    int num_males = 0;
    for (int i = 0; i < pride_sizes[pride_idx]; i++) {
        int idx = prides[pride_idx * LOA_PRIDE_SIZE + i];
        if (!genders[idx]) {
            males[num_males++] = idx;
        }
    }

    // Update male positions
    for (int i = 0; i < num_males; i++) {
        if (lion_idx != males[i]) continue;
        __global float *pos = population + lion_idx * dim;
        int num_visits = (int)(LOA_ROAMING_RATIO * pride_sizes[pride_idx]);
        for (int v = 0; v < num_visits; v++) {
            int target_idx = prides[pride_idx * LOA_PRIDE_SIZE + (int)(rand_float(&seeds[lion_idx]) * pride_sizes[pride_idx])];
            float d = 0.0f;
            for (int j = 0; j < dim; j++) {
                float diff = population[target_idx * dim + j] - pos[j];
                d += diff * diff;
            }
            d = sqrt(d);

            float norm = 0.0f;
            float direction[32]; // Assuming dim <= 32
            for (int j = 0; j < dim; j++) {
                direction[j] = population[target_idx * dim + j] - pos[j];
                norm += direction[j] * direction[j];
            }
            norm = norm > 0.0f ? sqrt(norm) : 1e-10f;
            for (int j = 0; j < dim; j++) {
                direction[j] /= norm;
            }

            float theta = rand_float(&seeds[lion_idx]) * M_PI / 3.0f - M_PI / 6.0f;
            float x = rand_float(&seeds[lion_idx]) * 2.0f * d;
            for (int j = 0; j < dim; j++) {
                pos[j] += x * direction[j];
                pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
            }

            // Placeholder fitness (evaluated on CPU)
            fitness[lion_idx] = INFINITY;
        }
    }
}

// Mating phase
__kernel void mating_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    __global int *prides,
    __global int *pride_sizes,
    __global uchar *genders,
    __global int *females,
    __global int *males,
    __global int *mating_females,
    __global float *temp_buffer,
    int pride_idx,
    int dim,
    int pop_size
) {
    int lid = get_global_id(0);
    if (lid >= pride_sizes[pride_idx]) return;

    int lion_idx = prides[pride_idx * LOA_PRIDE_SIZE + lid];

    // Collect females
    int num_females = 0;
    for (int i = 0; i < pride_sizes[pride_idx]; i++) {
        int idx = prides[pride_idx * LOA_PRIDE_SIZE + i];
        if (genders[idx]) {
            females[num_females++] = idx;
        }
    }

    // Select mating females
    int num_mating = (int)(LOA_MATING_RATIO * num_females);
    for (int i = 0; i < num_mating; i++) {
        mating_females[i] = females[(int)(rand_float(&seeds[lion_idx]) * num_females)];
    }

    // Collect males
    int num_males = 0;
    for (int i = 0; i < pride_sizes[pride_idx]; i++) {
        int idx = prides[pride_idx * LOA_PRIDE_SIZE + i];
        if (!genders[idx]) {
            males[num_males++] = idx;
        }
    }

    // Generate offspring
    for (int i = 0; i < num_mating; i++) {
        if (num_males == 0) continue;
        int female_idx = mating_females[i];
        int male_idx = males[(int)(rand_float(&seeds[lion_idx]) * num_males)];
        float beta = 0.4f + rand_float(&seeds[lion_idx]) * 0.2f;

        // Compute offspring
        for (int j = 0; j < dim; j++) {
            temp_buffer[j] = beta * population[female_idx * dim + j] + 
                             (1.0f - beta) * population[male_idx * dim + j];
            if (rand_float(&seeds[lion_idx]) < LOA_MUTATION_PROB) {
                temp_buffer[j] = bounds[j * 2] + rand_float(&seeds[lion_idx]) * (bounds[j * 2 + 1] - bounds[j * 2]);
            }
        }

        // Find worst individual (simplified: assume CPU handles replacement)
        // Placeholder fitness (evaluated on CPU)
        fitness[lion_idx] = INFINITY;
    }
}

// Nomad movement phase
__kernel void nomad_movement_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    __global int *nomads,
    int nomad_size,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= nomad_size) return;

    int lion_idx = nomads[gid];
    __global float *pos = population + lion_idx * dim;

    // Compute min/max fitness (simplified: assume CPU provides these)
    float min_fitness = INFINITY;
    float max_fitness = -INFINITY;
    for (int i = 0; i < pop_size; i++) {
        float f = fitness[i];
        min_fitness = min(min_fitness, f);
        max_fitness = max(max_fitness, f);
    }

    float pr = (fitness[lion_idx] - min_fitness) / (max_fitness - min_fitness + 1e-10f);
    for (int j = 0; j < dim; j++) {
        if (rand_float(&seeds[lion_idx]) > pr) {
            pos[j] = bounds[j * 2] + rand_float(&seeds[lion_idx]) * (bounds[j * 2 + 1] - bounds[j * 2]);
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    }

    // Placeholder fitness (evaluated on CPU)
    fitness[lion_idx] = INFINITY;
}

// Population control phase
__kernel void population_control_phase(
    __global int *pride_sizes,
    int num_prides
) {
    int gid = get_global_id(0);
    if (gid >= num_prides) return;

    if (pride_sizes[gid] > LOA_PRIDE_SIZE) {
        pride_sizes[gid] = LOA_PRIDE_SIZE;
    }
}
