// OpenCL kernel source for Flying Fox Optimization

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
    __global float *past_fitness,
    __global float *bounds,
    __global float *best_position,
    __global float *best_fitness,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    float local_best_fitness = INFINITY;
    int local_best_idx = gid;

    // Initialize position
    __global float *pos = population + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
    }

    // Placeholder: Assume objective_function is linked or inlined
    // For simplicity, use a dummy fitness (replace with actual objective function)
    float fit = 0.0f;
    for (int j = 0; j < dim; j++) fit += pos[j] * pos[j]; // Sphere function
    fitness[gid] = fit;
    past_fitness[gid] = fit;

    // Update best (simplified, assumes single work-group for reduction)
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (fit < local_best_fitness) {
        local_best_fitness = fit;
        for (int j = 0; j < dim; j++) best_position[j] = pos[j];
        *best_fitness = fit;
    }
}

// Fuzzy self-tuning for alpha and pa
__kernel void fuzzy_self_tuning(
    __global float *population,
    __global float *fitness,
    __global float *past_fitness,
    __global float *best_position,
    float best_fitness,
    float worst_fitness,
    __global float *alpha,
    __global float *pa,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    float deltamax = fabs(best_fitness - worst_fitness);
    float delta = fabs(best_fitness - fitness[gid]);
    float fi = deltamax > 0.0f ? (fitness[gid] - past_fitness[gid]) / deltamax : 0.0f;

    float deltas[3] = {0.2f * deltamax, 0.4f * deltamax, 0.6f * deltamax};
    float alpha_params[3] = {1.0f, 1.5f, 1.9f};
    float pa_params[3] = {0.5f, 0.85f, 0.99f};

    // Delta membership
    float delta_membership[3] = {0.0f, 0.0f, 0.0f};
    float fi_membership[3] = {0.0f, 1.0f - fabs(fi), 0.0f};

    if (delta < deltas[1]) {
        if (delta < deltas[0]) {
            delta_membership[0] = 1.0f;
        } else {
            delta_membership[0] = (deltas[1] - delta) / (deltas[1] - deltas[0]);
            delta_membership[1] = (delta - deltas[0]) / (deltas[1] - deltas[0]);
        }
    } else if (delta <= deltamax) {
        if (delta <= deltas[2]) {
            delta_membership[1] = (deltas[2] - delta) / (deltas[2] - deltas[1]);
            delta_membership[2] = (delta - deltas[1]) / (deltas[2] - deltas[1]);
        } else {
            delta_membership[2] = 1.0f;
        }
    }

    if (-1.0f <= fi && fi <= 1.0f) {
        if (fi < 0.0f) {
            fi_membership[0] = -fi;
            fi_membership[1] = 0.0f;
        } else if (fi > 0.0f) {
            fi_membership[2] = fi;
            fi_membership[1] = 0.0f;
        }
    }

    // Alpha rules
    float ruleno_alpha[3];
    ruleno_alpha[0] = fi_membership[0];
    ruleno_alpha[1] = max(fi_membership[1], max(delta_membership[0], delta_membership[1]));
    ruleno_alpha[2] = max(fi_membership[2], delta_membership[2]);
    float alpha_sum = ruleno_alpha[0] + ruleno_alpha[1] + ruleno_alpha[2];
    alpha[gid] = alpha_sum > 0.0f ? (ruleno_alpha[0] * alpha_params[0] +
                                     ruleno_alpha[1] * alpha_params[1] +
                                     ruleno_alpha[2] * alpha_params[2]) / alpha_sum : 1.0f;

    // Pa rules
    float ruleno_pa[3];
    ruleno_pa[0] = max(fi_membership[2], delta_membership[2]);
    ruleno_pa[1] = max(fi_membership[1], delta_membership[0]);
    ruleno_pa[2] = max(fi_membership[0], delta_membership[1]);
    float pa_sum = ruleno_pa[0] + ruleno_pa[1] + ruleno_pa[2];
    pa[gid] = pa_sum > 0.0f ? (ruleno_pa[0] * pa_params[0] +
                               ruleno_pa[1] * pa_params[1] +
                               ruleno_pa[2] * pa_params[2]) / pa_sum : 0.85f;
}

// Update position
__kernel void update_position(
    __global float *population,
    __global float *fitness,
    __global float *past_fitness,
    __global float *best_position,
    float best_fitness,
    float worst_fitness,
    __global float *alpha,
    __global float *pa,
    __global float *bounds,
    __global float *survival_list,
    __global float *survival_fitness,
    __global int *survival_count,
    __global uint *seeds,
    int dim,
    int pop_size,
    int surv_list_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    float deltamax = fabs(best_fitness - worst_fitness);
    float deltas[3] = {0.2f * deltamax, 0.4f * deltamax, 0.6f * deltamax};
    __global float *pos = population + gid * dim;

    if (fabs(fitness[gid] - best_fitness) > (deltas[0] * 0.5f)) {
        // Global movement
        for (int j = 0; j < dim; j++) {
            pos[j] += alpha[gid] * rand_float(&seeds[gid]) * (best_position[j] - pos[j]);
            pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }
    } else {
        // Local search
        int A[2];
        A[0] = (gid + 1) % pop_size;
        A[1] = (gid + 2) % pop_size; // Simplified neighbor selection

        for (int j = 0; j < dim; j++) {
            float step = rand_float(&seeds[gid]) * (best_position[j] - pos[j]) +
                         rand_float(&seeds[gid]) * (population[A[0] * dim + j] - population[A[1] * dim + j]);
            if (j == (int)(rand_float(&seeds[gid]) * dim) || rand_float(&seeds[gid]) >= pa[gid]) {
                pos[j] += step;
                pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
            }
        }
    }

    // Compute new fitness (dummy sphere function)
    float new_fitness = 0.0f;
    for (int j = 0; j < dim; j++) new_fitness += pos[j] * pos[j];
    float old_fitness = fitness[gid];
    fitness[gid] = new_fitness;
    if (new_fitness < old_fitness) {
        past_fitness[gid] = old_fitness;
    }

    // Update survival list (simplified)
    if (fabs(new_fitness - best_fitness) > deltas[2] && *survival_count < surv_list_size) {
        int idx = atomic_add(survival_count, 1);
        if (idx < surv_list_size) {
            for (int j = 0; j < dim; j++) survival_list[idx * dim + j] = pos[j];
            survival_fitness[idx] = new_fitness;
        }
    }
}

// Crossover
__kernel void crossover(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0) * 2;
    if (gid >= pop_size) return;

    int parent1 = (int)(rand_float(&seeds[gid]) * pop_size);
    int parent2 = (int)(rand_float(&seeds[gid]) * pop_size);
    if (parent1 == parent2) return;

    __global float *off1 = population + gid * dim;
    __global float *off2 = population + (gid + 1) * dim;
    __global float *p1 = population + parent1 * dim;
    __global float *p2 = population + parent2 * dim;

    for (int j = 0; j < dim; j++) {
        float L = rand_float(&seeds[gid]);
        off1[j] = L * p1[j] + (1.0f - L) * p2[j];
        off2[j] = L * p2[j] + (1.0f - L) * p1[j];
        off1[j] = clamp(off1[j], bounds[j * 2], bounds[j * 2 + 1]);
        off2[j] = clamp(off2[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Update fitness (dummy)
    float fit1 = 0.0f, fit2 = 0.0f;
    for (int j = 0; j < dim; j++) {
        fit1 += off1[j] * off1[j];
        fit2 += off2[j] * off2[j];
    }
    fitness[gid] = fit1;
    fitness[gid + 1] = fit2;
}

// Suffocation phase
__kernel void suffocation(
    __global float *population,
    __global float *fitness,
    __global float *past_fitness,
    __global float *survival_list,
    __global float *survival_fitness,
    int survival_count,
    __global float *bounds,
    float best_fitness,
    __global uint *seeds,
    int dim,
    int pop_size,
    int surv_list_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    if (fitness[gid] == best_fitness) {
        float p_death = 0.5f; // Simplified
        if (rand_float(&seeds[gid]) < p_death) {
            if (survival_count >= 2) {
                int idx = (int)(rand_float(&seeds[gid]) * survival_count);
                for (int j = 0; j < dim; j++) {
                    population[gid * dim + j] = survival_list[idx * dim + j];
                }
                fitness[gid] = survival_fitness[idx];
                past_fitness[gid] = fitness[gid];
            } else {
                for (int j = 0; j < dim; j++) {
                    float lower = bounds[j * 2];
                    float upper = bounds[j * 2 + 1];
                    population[gid * dim + j] = lower + rand_float(&seeds[gid]) * (upper - lower);
                }
                float fit = 0.0f;
                for (int j = 0; j < dim; j++) fit += population[gid * dim + j] * population[gid * dim + j];
                fitness[gid] = fit;
                past_fitness[gid] = fit;
            }
        }
    }
}
