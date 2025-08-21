#define MFA_PERSONAL_COEFF 1.0f
#define MFA_GLOBAL_COEFF1 1.5f
#define MFA_GLOBAL_COEFF2 1.5f
#define MFA_DISTANCE_COEFF 2.0f
#define MFA_NUM_OFFSPRING 20
#define MFA_NUM_MUTANTS 1
#define MFA_MUTATION_RATE 0.01f

// Random number generator (XOR-shift)
inline float rand_float(uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}

// Initialize population
__kernel void initialize_population(
    __global float *male_population,
    __global float *female_population,
    __global float *male_fitness,
    __global float *female_fitness,
    __global float *bounds,
    __global float *best_position,
    __global float *best_fitness,
    __global uint *seeds,
    int dim,
    int half_pop
) {
    int gid = get_global_id(0);
    if (gid >= half_pop) return;

    // Initialize male position
    __global float *male_pos = male_population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        male_pos[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
    }

    // Initialize female position
    __global float *female_pos = female_population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        female_pos[j] = lower + rand_float(&seeds[gid + half_pop]) * (upper - lower);
    }

    // Placeholder fitness (to be evaluated on CPU)
    male_fitness[gid] = INFINITY;
    female_fitness[gid] = INFINITY;
}

// Update male population
__kernel void update_males(
    __global float *male_population,
    __global float *male_fitness,
    __global float *male_velocities,
    __global float *best_male_positions,
    __global float *best_position,
    __global float *best_fitness,
    __global float *bounds,
    __global uint *seeds,
    float inertia_weight,
    float vel_max,
    float vel_min,
    float nuptial_dance,
    int dim,
    int half_pop
) {
    int gid = get_global_id(0);
    if (gid >= half_pop) return;

    __global float *pos = male_population + gid * dim;
    __global float *vel = male_velocities + gid * dim;
    __global float *best_pos = best_male_positions + gid * dim;
    float current_fitness = male_fitness[gid];

    if (current_fitness > *best_fitness) {
        for (int j = 0; j < dim; j++) {
            float rpbest = best_pos[j] - pos[j];
            float rgbest = best_position[j] - pos[j];
            vel[j] = (inertia_weight * vel[j] +
                      MFA_PERSONAL_COEFF * exp(-MFA_DISTANCE_COEFF * rpbest * rpbest) * rpbest +
                      MFA_GLOBAL_COEFF1 * exp(-MFA_DISTANCE_COEFF * rgbest * rgbest) * rgbest);
            vel[j] = clamp(vel[j], vel_min, vel_max);
            pos[j] += vel[j];
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    } else {
        for (int j = 0; j < dim; j++) {
            vel[j] = inertia_weight * vel[j] + nuptial_dance * (rand_float(&seeds[gid]) - 0.5f);
            vel[j] = clamp(vel[j], vel_min, vel_max);
            pos[j] += vel[j];
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    }

    // Update best male position if improved
    male_fitness[gid] = INFINITY; // Placeholder for CPU evaluation
}

// Update female population
__kernel void update_females(
    __global float *female_population,
    __global float *female_fitness,
    __global float *female_velocities,
    __global float *male_population,
    __global float *male_fitness,
    __global float *bounds,
    __global uint *seeds,
    float inertia_weight,
    float random_flight,
    float vel_max,
    float vel_min,
    int dim,
    int half_pop
) {
    int gid = get_global_id(0);
    if (gid >= half_pop) return;

    __global float *female_pos = female_population + gid * dim;
    __global float *male_pos = male_population + gid * dim;
    __global float *vel = female_velocities + gid * dim;
    float female_fit = female_fitness[gid];
    float male_fit = male_fitness[gid];

    if (female_fit > male_fit) {
        for (int j = 0; j < dim; j++) {
            float rmf = male_pos[j] - female_pos[j];
            vel[j] = (inertia_weight * vel[j] +
                      MFA_GLOBAL_COEFF2 * exp(-MFA_DISTANCE_COEFF * rmf * rmf) * rmf);
            vel[j] = clamp(vel[j], vel_min, vel_max);
            female_pos[j] += vel[j];
            female_pos[j] = clamp(female_pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    } else {
        for (int j = 0; j < dim; j++) {
            vel[j] = inertia_weight * vel[j] + random_flight * (rand_float(&seeds[gid + half_pop]) - 0.5f);
            vel[j] = clamp(vel[j], vel_min, vel_max);
            female_pos[j] += vel[j];
            female_pos[j] = clamp(female_pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    }

    female_fitness[gid] = INFINITY; // Placeholder for CPU evaluation
}

// Mating phase
__kernel void mating_phase(
    __global float *male_population,
    __global float *female_population,
    __global float *offspring_population,
    __global float *offspring_fitness,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int half_pop
) {
    int gid = get_global_id(0);
    if (gid >= MFA_NUM_OFFSPRING / 2) return;

    __global float *male_pos = male_population + gid * dim;
    __global float *female_pos = female_population + gid * dim;
    __global float *off1 = offspring_population + (2 * gid) * dim;
    __global float *off2 = offspring_population + (2 * gid + 1) * dim;

    for (int j = 0; j < dim; j++) {
        float L = rand_float(&seeds[gid]);
        off1[j] = L * male_pos[j] + (1.0f - L) * female_pos[j];
        off2[j] = L * female_pos[j] + (1.0f - L) * male_pos[j];
        off1[j] = clamp(off1[j], bounds[j * 2], bounds[j * 2 + 1]);
        off2[j] = clamp(off2[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    offspring_fitness[2 * gid] = INFINITY;
    offspring_fitness[2 * gid + 1] = INFINITY;
}

// Mutation phase
__kernel void mutation_phase(
    __global float *male_population,
    __global float *female_population,
    __global float *mutant_population,
    __global float *mutant_fitness,
    __global float *bounds,
    __global uint *seeds,
    float mutation_rate,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= MFA_NUM_MUTANTS) return;

    int idx = (int)(rand_float(&seeds[gid]) * pop_size);
    __global float *src_pos = (idx < pop_size / 2) ? (male_population + idx * dim) : (female_population + (idx - pop_size / 2) * dim);
    __global float *mutant_pos = mutant_population + gid * dim;

    int n_mu = (int)(mutation_rate * dim);
    for (int j = 0; j < dim; j++) {
        mutant_pos[j] = src_pos[j];
    }

    for (int m = 0; m < n_mu; m++) {
        int j = (int)(rand_float(&seeds[gid]) * dim);
        float sigma = 0.1f * (bounds[j * 2 + 1] - bounds[j * 2]);
        mutant_pos[j] += sigma * (rand_float(&seeds[gid]) - 0.5f);
        mutant_pos[j] = clamp(mutant_pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    mutant_fitness[gid] = INFINITY;
}
