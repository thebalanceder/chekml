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

    // Placeholder fitness (to be evaluated on CPU)
    fitness[gid] = INFINITY;
}

// Exploration phase
__kernel void exploration_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global float *temp_position,
    __global float *y,
    __global float *z,
    __global uint *seeds,
    float pCR,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    __global float *temp_pos = temp_position + gid * dim;
    __global float *y_vec = y + gid * dim;
    __global float *z_vec = z + gid * dim;

    // Copy current position to temp
    for (int j = 0; j < dim; j++) {
        temp_pos[j] = pos[j];
    }

    // Generate indices for other pumas (simplified permutation)
    int a = (gid + 1) % pop_size;
    int b = (gid + 2) % pop_size;
    int c = (gid + 3) % pop_size;
    int d = (gid + 4) % pop_size;
    int e = (gid + 5) % pop_size;
    int f = (gid + 6) % pop_size;

    float G = 2.0f * rand_float(&seeds[gid]) - 1.0f;
    if (rand_float(&seeds[gid]) < 0.5f) {
        for (int j = 0; j < dim; j++) {
            float lower = bounds[j * 2];
            float upper = bounds[j * 2 + 1];
            y_vec[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
        }
    } else {
        __global float *pos_a = population + a * dim;
        __global float *pos_b = population + b * dim;
        __global float *pos_c = population + c * dim;
        __global float *pos_d = population + d * dim;
        __global float *pos_e = population + e * dim;
        __global float *pos_f = population + f * dim;
        for (int j = 0; j < dim; j++) {
            y_vec[j] = pos_a[j] + G * (pos_a[j] - pos_b[j]) +
                       G * ((pos_a[j] - pos_b[j]) - (pos_c[j] - pos_d[j]) +
                            (pos_c[j] - pos_d[j]) - (pos_e[j] - pos_f[j]));
        }
    }

    // Enforce bounds on y
    for (int j = 0; j < dim; j++) {
        float yj = y_vec[j];
        float lb = bounds[j * 2];
        float ub = bounds[j * 2 + 1];
        y_vec[j] = clamp(yj, lb, ub);
    }

    // Crossover to create z
    for (int j = 0; j < dim; j++) {
        z_vec[j] = temp_pos[j];
    }
    int j0 = (int)(rand_float(&seeds[gid]) * dim);
    for (int j = 0; j < dim; j++) {
        if (j == j0 || rand_float(&seeds[gid]) <= pCR) {
            z_vec[j] = y_vec[j];
        }
    }

    // Copy z back to population (fitness evaluated on CPU)
    for (int j = 0; j < dim; j++) {
        pos[j] = z_vec[j];
    }
    fitness[gid] = INFINITY;
}

// Exploitation phase
__kernel void exploitation_phase(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global float *best_position,
    __global float *beta2,
    __global float *w,
    __global float *v,
    __global float *F1,
    __global float *F2,
    __global float *S1,
    __global float *S2,
    __global float *VEC,
    __global float *Xatack,
    __global float *mbest,
    __global uint *seeds,
    float q_probability,
    float beta,
    int iter,
    int max_iter,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    __global float *beta2_vec = beta2 + gid * dim;
    __global float *w_vec = w + gid * dim;
    __global float *v_vec = v + gid * dim;
    __global float *F1_vec = F1 + gid * dim;
    __global float *F2_vec = F2 + gid * dim;
    __global float *S1_vec = S1 + gid * dim;
    __global float *S2_vec = S2 + gid * dim;
    __global float *VEC_vec = VEC + gid * dim;
    __global float *Xatack_vec = Xatack + gid * dim;

    float T = (float)max_iter;
    float beta1 = 2.0f * rand_float(&seeds[gid]);
    float rand_val = rand_float(&seeds[gid]);

    for (int j = 0; j < dim; j++) {
        beta2_vec[j] = rand_float(&seeds[gid]) * 2.0f - 1.0f;
        w_vec[j] = rand_float(&seeds[gid]) * 2.0f - 1.0f;
        v_vec[j] = rand_float(&seeds[gid]) * 2.0f - 1.0f;
        F1_vec[j] = (rand_float(&seeds[gid]) * 2.0f - 1.0f) * exp(2.0f - iter * (2.0f / T));
        F2_vec[j] = w_vec[j] * v_vec[j] * v_vec[j] * cos(2.0f * rand_val * w_vec[j]);
        S1_vec[j] = 2.0f * rand_float(&seeds[gid]) - 1.0f + (rand_float(&seeds[gid]) * 2.0f - 1.0f);
        S2_vec[j] = F1_vec[j] * (2.0f * rand_float(&seeds[gid]) - 1.0f) * pos[j] +
                    F2_vec[j] * (1.0f - (2.0f * rand_float(&seeds[gid]) - 1.0f)) * best_position[j];
        VEC_vec[j] = S2_vec[j] / S1_vec[j];
    }

    if (rand_val <= 0.5f) {
        for (int j = 0; j < dim; j++) {
            Xatack_vec[j] = VEC_vec[j];
        }
        if (rand_float(&seeds[gid]) > q_probability) {
            int r1 = (int)(rand_float(&seeds[gid]) * pop_size);
            __global float *pos_r1 = population + r1 * dim;
            for (int j = 0; j < dim; j++) {
                pos[j] = best_position[j] + beta1 * exp(beta2_vec[j]) * (pos_r1[j] - pos[j]);
            }
        } else {
            for (int j = 0; j < dim; j++) {
                pos[j] = beta1 * Xatack_vec[j] - best_position[j];
            }
        }
    } else {
        int r1 = 1 + (int)(rand_float(&seeds[gid]) * (pop_size - 1));
        float sign = rand_float(&seeds[gid]) < 0.5f ? 1.0f : -1.0f;
        float denom = 1.0f + (beta * rand_float(&seeds[gid]));
        __global float *pos_r1 = population + r1 * dim;
        for (int j = 0; j < dim; j++) {
            pos[j] = (mbest[j] * pos_r1[j] - sign * pos[j]) / denom;
        }
    }

    // Enforce bounds
    for (int j = 0; j < dim; j++) {
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    fitness[gid] = INFINITY;
}
