#define SOA_MU_MAX 0.9f
#define SOA_MU_MIN 0.05f
#define W_MAX_SOA 0.9f
#define W_MIN_SOA 0.2f
#define NUM_REGIONS 3

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
    __global float *pbest_s,
    __global float *pbest_fun,
    __global float *e_t_1,
    __global float *e_t_2,
    __global float *f_t_1,
    __global float *f_t_2,
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
        pbest_s[gid * dim + j] = pos[j];
        e_t_1[gid * dim + j] = pos[j];
        e_t_2[gid * dim + j] = pos[j];
    }

    // Placeholder fitness
    fitness[gid] = INFINITY;
    pbest_fun[gid] = INFINITY;
    f_t_1[gid] = INFINITY;
    f_t_2[gid] = INFINITY;
}

// Update solutions
__kernel void update_solutions(
    __global float *population,
    __global float *fitness,
    __global float *pbest_s,
    __global float *pbest_fun,
    __global float *lbest_s,
    __global float *lbest_fun,
    __global float *e_t_1,
    __global float *e_t_2,
    __global float *f_t_1,
    __global float *f_t_2,
    __global float *bounds,
    __global uint *seeds,
    __global int *start_reg,
    __global int *end_reg,
    __global int *size_reg,
    __global float *rmax,
    __global float *rmin,
    float weight,
    float mu,
    int dim,
    int pop_size,
    int num_regions
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    // Determine region
    int region = 0;
    for (int r = 0; r < num_regions; r++) {
        if (gid >= start_reg[r] && gid < end_reg[r]) {
            region = r;
            break;
        }
    }

    if (size_reg[region] <= 0) return;

    __global float *pos = population + gid * dim;
    float current_fitness = fitness[gid];

    // Find worst in region
    int worst_idx = start_reg[region];
    float worst_fitness = fitness[worst_idx];
    for (int i = start_reg[region]; i < end_reg[region]; i++) {
        if (fitness[i] > worst_fitness) {
            worst_fitness = fitness[i];
            worst_idx = i;
        }
    }

    // Compute exploration term
    float en_temp[32]; // Assuming dim <= 32 for simplicity
    if (size_reg[region] >= 2) {
        int rand_en = 1 + (int)(rand_float(&seeds[gid]) * (size_reg[region] - 2));
        if (rand_en >= size_reg[region]) rand_en = size_reg[region] - 1;
        int rand_idx = start_reg[region] + rand_en;
        for (int j = 0; j < dim; j++) {
            en_temp[j] = weight * fabs(population[worst_idx * dim + j] - population[rand_idx * dim + j]);
        }
    } else {
        for (int j = 0; j < dim; j++) {
            en_temp[j] = 0.0f;
        }
    }

    // Compute mu_s
    float mu_s[32];
    for (int j = 0; j < dim; j++) {
        mu_s[j] = mu + (1.0f - mu) * rand_float(&seeds[gid]);
        if (mu_s[j] <= 0.0f || mu_s[j] >= 1.0f) mu_s[j] = mu;
    }

    // Compute directions
    float x_pdirect[32], x_ldirect1[32], x_ldirect2[32], x_tdirect[32];
    for (int j = 0; j < dim; j++) {
        x_pdirect[j] = pbest_s[gid * dim + j] - pos[j];
        x_ldirect1[j] = (lbest_fun[region] < current_fitness) ? lbest_s[region * dim + j] - pos[j] : 0.0f;
        x_ldirect2[j] = (fitness[worst_idx] < current_fitness) ? population[worst_idx * dim + j] - pos[j] : 0.0f;
    }

    // Temporal direction
    float f_values[3] = {f_t_2[gid], f_t_1[gid], current_fitness};
    __global float *e_values[3] = {e_t_2 + gid * dim, e_t_1 + gid * dim, pos};
    int order_idx[3] = {0, 1, 2};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2 - i; j++) {
            if (f_values[order_idx[j]] > f_values[order_idx[j + 1]]) {
                int temp = order_idx[j];
                order_idx[j] = order_idx[j + 1];
                order_idx[j + 1] = temp;
            }
        }
    }
    for (int j = 0; j < dim; j++) {
        x_tdirect[j] = e_values[order_idx[0]][j] - e_values[order_idx[2]][j];
    }

    // Compute direction signs
    int flag_direct[4] = {1, 1, (lbest_fun[region] < current_fitness) ? 1 : 0, 
                          (fitness[worst_idx] < current_fitness) ? 1 : 0};
    float x_signs[4][32];
    for (int j = 0; j < dim; j++) {
        x_signs[0][j] = (x_tdirect[j] > 0) ? 1 : (x_tdirect[j] < 0) ? -1 : 0;
        x_signs[1][j] = (x_pdirect[j] > 0) ? 1 : (x_pdirect[j] < 0) ? -1 : 0;
        x_signs[2][j] = (x_ldirect1[j] > 0) ? 1 : (x_ldirect1[j] < 0) ? -1 : 0;
        x_signs[3][j] = (x_ldirect2[j] > 0) ? 1 : (x_ldirect2[j] < 0) ? -1 : 0;
    }

    int select_sign[4], num_sign = 0;
    for (int i = 0; i < 4; i++) {
        if (flag_direct[i] > 0) select_sign[num_sign++] = i;
    }

    float num_pone[32], num_none[32];
    for (int j = 0; j < dim; j++) {
        num_pone[j] = 0.0f;
        num_none[j] = 0.0f;
    }
    for (int i = 0; i < num_sign; i++) {
        for (int j = 0; j < dim; j++) {
            num_pone[j] += (fabs(x_signs[select_sign[i]][j]) + x_signs[select_sign[i]][j]) / 2.0f;
            num_none[j] += (fabs(x_signs[select_sign[i]][j]) - x_signs[select_sign[i]][j]) / 2.0f;
        }
    }

    float x_direct[32];
    for (int j = 0; j < dim; j++) {
        float prob_pone = num_sign > 0 ? num_pone[j] / num_sign : 0.5f;
        float prob_none = num_sign > 0 ? (num_pone[j] + num_none[j]) / num_sign : 1.0f;
        float rand_roulette = rand_float(&seeds[gid]);
        x_direct[j] = (rand_roulette <= prob_pone) ? 1.0f : (rand_roulette <= prob_none) ? -1.0f : 0.0f;

        // Adjust direction based on bounds
        if (pos[j] > bounds[j * 2 + 1]) x_direct[j] = -1.0f;
        if (pos[j] < bounds[j * 2]) x_direct[j] = 1.0f;
        if (x_direct[j] == 0.0f) {
            x_direct[j] = (rand_float(&seeds[gid]) < 0.5f) ? 1.0f : -1.0f;
        }
    }

    // Compute step
    for (int j = 0; j < dim; j++) {
        float r_temp = x_direct[j] * (en_temp[j] * sqrt(-2.0f * log(mu_s[j])));
        r_temp = max(rmin[j], min(rmax[j], r_temp));
        pos[j] = pos[j] + r_temp;
        pos[j] = clamp(pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    // Placeholder fitness
    fitness[gid] = INFINITY;
}
