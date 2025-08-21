#define KHO_VF 0.02f
#define KHO_DMAX 0.005f
#define KHO_NMAX 0.01f
#define KHO_CROSSOVER_RATE 0.8f
#define KHO_CROSSOVER_SCALE 0.2f
#define KHO_INERTIA_MIN 0.1f
#define KHO_INERTIA_MAX 0.8f
#define KHO_NEIGHBOR_LIMIT 4
#define KHO_SENSE_DISTANCE_FACTOR 5.0f
#define KHO_MAX_ITER 500

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

// Movement-induced phase
__kernel void movement_induced_phase(
    __global float *population,
    __global float *fitness,
    __global float *best_position,
    __global float *bounds,
    __global uint *seeds,
    __global float *N,
    float w,
    float Kw_Kgb,
    int dim,
    int pop_size,
    float best_fitness,
    int iteration
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos_i = population + gid * dim;
    __global float *N_i = N + gid * dim;
    float fit_i = fitness[gid];
    float inv_Kw_Kgb = Kw_Kgb > 1e-10f ? 1.0f / Kw_Kgb : 0.0f;
    float ds = 0.0f;

    // Compute distances to other krill
    float R[32]; // Assuming pop_size <= 32 for simplicity
    for (int n = 0; n < pop_size; n++) {
        R[n] = 0.0f;
        __global float *pos_n = population + n * dim;
        for (int j = 0; j < dim; j++) {
            float diff = pos_n[j] - pos_i[j];
            R[n] += diff * diff;
        }
        R[n] = sqrt(R[n]);
        ds += R[n];
    }
    ds /= (pop_size * KHO_SENSE_DISTANCE_FACTOR);

    // Compute alpha_b (global best influence)
    float Rgb[32]; // Assuming dim <= 32
    float norm_Rgb = 0.0f;
    for (int j = 0; j < dim; j++) {
        Rgb[j] = best_position[j] - pos_i[j];
        norm_Rgb += Rgb[j] * Rgb[j];
    }
    norm_Rgb = sqrt(norm_Rgb);

    float alpha_b = 0.0f;
    if (best_fitness < fit_i && norm_Rgb > 1e-10f) {
        alpha_b = -2.0f * (1.0f + rand_float(&seeds[gid]) * ((float)iteration / KHO_MAX_ITER)) *
                  (best_fitness - fit_i) * inv_Kw_Kgb / norm_Rgb;
    }

    // Compute alpha_n (neighbor influence)
    float alpha_n = 0.0f;
    int nn = 0;
    for (int n = 0; n < pop_size && nn < KHO_NEIGHBOR_LIMIT; n++) {
        if (R[n] < ds && n != gid) {
            float fit_n = fitness[n];
            if (fit_i != fit_n) {
                alpha_n -= (fit_n - fit_i) * inv_Kw_Kgb / (R[n] + 1e-10f);
            }
            nn++;
        }
    }

    // Update induced motion
    for (int j = 0; j < dim; j++) {
        float motion = alpha_b * Rgb[j];
        nn = 0;
        for (int n = 0; n < pop_size && nn < KHO_NEIGHBOR_LIMIT; n++) {
            if (R[n] < ds && n != gid) {
                __global float *pos_n = population + n * dim;
                motion += alpha_n * (pos_n[j] - pos_i[j]);
                nn++;
            }
        }
        N_i[j] = w * N_i[j] + KHO_NMAX * motion;
    }
}

// Foraging motion phase
__kernel void foraging_motion_phase(
    __global float *population,
    __global float *fitness,
    __global float *Xf,
    __global float *local_best_pos,
    __global float *local_best_fitness,
    __global uint *seeds,
    __global float *F,
    float w,
    float Kw_Kgb,
    int dim,
    int pop_size,
    float Kf,
    int iteration
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos_i = population + gid * dim;
    __global float *F_i = F + gid * dim;
    float fit_i = fitness[gid];
    __global float *best_pos_i = local_best_pos + gid * dim;
    float best_fit_i = local_best_fitness[gid];
    float inv_Kw_Kgb = Kw_Kgb > 1e-10f ? 1.0f / Kw_Kgb : 0.0f;

    // Compute distances
    float Rf[32]; // Assuming dim <= 32
    float Rib[32];
    float norm_Rf = 0.0f;
    float norm_Rib = 0.0f;
    for (int j = 0; j < dim; j++) {
        Rf[j] = Xf[j] - pos_i[j];
        Rib[j] = best_pos_i[j] - pos_i[j];
        norm_Rf += Rf[j] * Rf[j];
        norm_Rib += Rib[j] * Rib[j];
    }
    norm_Rf = sqrt(norm_Rf);
    norm_Rib = sqrt(norm_Rib);

    // Compute Beta_f (food attraction)
    float Beta_f = 0.0f;
    if (Kf < fit_i && norm_Rf > 1e-10f) {
        Beta_f = -2.0f * (1.0f - (float)iteration / KHO_MAX_ITER) *
                 (Kf - fit_i) * inv_Kw_Kgb / norm_Rf;
    }

    // Compute Beta_b (best position attraction)
    float Beta_b = 0.0f;
    if (best_fit_i < fit_i && norm_Rib > 1e-10f) {
        Beta_b = -(best_fit_i - fit_i) * inv_Kw_Kgb / norm_Rib;
    }

    // Update foraging motion
    for (int j = 0; j < dim; j++) {
        F_i[j] = w * F_i[j] + KHO_VF * (Beta_f * Rf[j] + Beta_b * Rib[j]);
    }
}

// Physical diffusion phase
__kernel void physical_diffusion_phase(
    __global float *fitness,
    __global uint *seeds,
    __global float *D,
    int iteration,
    float Kw_Kgb,
    int dim,
    int pop_size,
    float best_fitness
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *D_i = D + gid * dim;
    float fit_i = fitness[gid];
    float inv_Kw_Kgb = Kw_Kgb > 1e-10f ? 1.0f / Kw_Kgb : 0.0f;
    float scale = KHO_DMAX * (1.0f - (float)iteration / KHO_MAX_ITER);
    float diff = (fit_i - best_fitness) * inv_Kw_Kgb;
    float diffusion = scale * (rand_float(&seeds[gid]) + diff);

    for (int j = 0; j < dim; j++) {
        D_i[j] = diffusion * (2.0f * rand_float(&seeds[gid]) - 1.0f);
    }
}

// Crossover phase
__kernel void crossover_phase(
    __global float *population,
    __global float *fitness,
    __global uint *seeds,
    float Kw_Kgb,
    int dim,
    int pop_size,
    float best_fitness
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos_i = population + gid * dim;
    float fit_i = fitness[gid];
    float inv_Kw_Kgb = Kw_Kgb > 1e-10f ? 1.0f / Kw_Kgb : 0.0f;
    float C_rate = KHO_CROSSOVER_RATE + KHO_CROSSOVER_SCALE * (fit_i - best_fitness) * inv_Kw_Kgb;
    int partner = (int)(rand_float(&seeds[gid]) * (pop_size - 1));
    if (partner >= gid) partner++;

    __global float *pos_partner = population + partner * dim;
    for (int j = 0; j < dim; j++) {
        if (rand_float(&seeds[gid]) < C_rate) {
            pos_i[j] = pos_partner[j];
        }
    }
}

// Update positions and enforce bounds
__kernel void update_positions(
    __global float *population,
    __global float *N,
    __global float *F,
    __global float *D,
    __global float *bounds,
    __global float *best_position,
    __global uint *seeds,
    float Dt,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos_i = population + gid * dim;
    __global float *N_i = N + gid * dim;
    __global float *F_i = F + gid * dim;
    __global float *D_i = D + gid * dim;

    for (int j = 0; j < dim; j++) {
        pos_i[j] += Dt * (N_i[j] + F_i[j] + D_i[j]);
        float lb = bounds[j * 2];
        float ub = bounds[j * 2 + 1];
        if (pos_i[j] < lb) {
            float A = rand_float(&seeds[gid]);
            pos_i[j] = A * lb + (1.0f - A) * best_position[j];
        } else if (pos_i[j] > ub) {
            float B = rand_float(&seeds[gid]);
            pos_i[j] = B * ub + (1.0f - B) * best_position[j];
        }
    }
}
