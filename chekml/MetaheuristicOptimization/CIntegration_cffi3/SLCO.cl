#define SLCO_N_TEAMS 10
#define SLCO_N_MAIN_PLAYERS 5
#define SLCO_N_RESERVE_PLAYERS 3

inline float rand_float(__global uint *seed) {
    *seed = (*seed * 1103515245u + 12345u) & 0x7fffffffu;
    return (float)(*seed) / (float)0x7fffffffu;
}

inline float gaussian_random(__global uint *seed, float mean, float stddev) {
    float u1 = rand_float(seed);
    float u2 = rand_float(seed);
    float z0 = sqrt(-2.0f * log(u1)) * cos(2.0f * 3.141592653589793f * u2);
    return mean + stddev * z0;
}

// Custom atomic add for float using compare-and-swap
inline void atomic_add_float(__global float *addr, float val) {
    union {
        float f;
        uint i;
    } old_val, new_val;

    do {
        old_val.f = *addr;
        new_val.f = old_val.f + val;
    } while (atomic_cmpxchg((volatile __global uint *)addr, old_val.i, new_val.i) != old_val.i);
}

__kernel void initialize_league(__global float *positions, __global float *costs,
                                __global float *bounds, __global uint *seeds,
                                int dim, int total_players) {
    int gid = get_global_id(0);
    if (gid >= total_players) return;

    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        positions[gid * dim + j] = lower + (upper - lower) * rand_float(&seeds[gid]);
    }
    costs[gid] = 0.0f; // Will be set by CPU
}

__kernel void takhsis(__global float *positions, __global float *costs,
                      __global int *teams, __global float *team_total_costs,
                      int n_teams, int n_main, int n_reserve, int dim) {
    int gid = get_global_id(0);
    int total_players = n_teams * (n_main + n_reserve);
    if (gid >= total_players) return;

    int team = teams[gid];
    if (team < n_teams) {
        atomic_add_float(&team_total_costs[team], costs[gid]);
    }
}

__kernel void winner_function_main(__global float *positions, __global float *costs,
                                  __global int *teams, __global float *bounds,
                                  __global uint *seeds, float alpha, float beta,
                                  int winner_team, int n_teams, int n_main, int dim) {
    int gid = get_global_id(0);
    int total_players = n_teams * (n_main + SLCO_N_RESERVE_PLAYERS);
    if (gid >= total_players || teams[gid] != winner_team || (gid % (n_main + SLCO_N_RESERVE_PLAYERS)) >= n_main) return;

    for (int j = 0; j < dim; j++) {
        int r1 = (int)(rand_float(&seeds[gid]) * n_main);
        int r2 = (int)(rand_float(&seeds[gid]) * n_main);
        while (r1 == r2 || r1 == (gid % (n_main + SLCO_N_RESERVE_PLAYERS)) || r2 == (gid % (n_main + SLCO_N_RESERVE_PLAYERS))) {
            r1 = (int)(rand_float(&seeds[gid]) * n_main);
            r2 = (int)(rand_float(&seeds[gid]) * n_main);
        }
        int r1_idx = winner_team * (n_main + SLCO_N_RESERVE_PLAYERS) + r1;
        int r2_idx = winner_team * (n_main + SLCO_N_RESERVE_PLAYERS) + r2;

        float pos = positions[gid * dim + j];
        pos += alpha * (positions[r1_idx * dim + j] - positions[r2_idx * dim + j]);
        pos = max(bounds[j * 2], min(bounds[j * 2 + 1], pos));
        positions[gid * dim + j] = pos;
    }
}

__kernel void winner_function_reserve(__global float *positions, __global float *costs,
                                     __global int *teams, __global float *bounds,
                                     __global uint *seeds, float beta,
                                     int winner_team, int n_teams, int n_main,
                                     int n_reserve, int dim) {
    int gid = get_global_id(0);
    int total_players = n_teams * (n_main + n_reserve);
    if (gid >= total_players || teams[gid] != winner_team || (gid % (n_main + n_reserve)) < n_main) return;

    for (int j = 0; j < dim; j++) {
        float pos = positions[gid * dim + j];
        pos += beta * rand_float(&seeds[gid]) * (bounds[j * 2 + 1] - bounds[j * 2]);
        pos = max(bounds[j * 2], min(bounds[j * 2 + 1], pos));
        positions[gid * dim + j] = pos;
    }
}

__kernel void loser_function(__global float *positions, __global float *costs,
                             __global int *teams, __global float *bounds,
                             __global uint *seeds, float alpha,
                             int loser_team, int n_teams, int n_main,
                             int n_reserve, int dim, float iteration, int max_iter) {
    int gid = get_global_id(0);
    int total_players = n_teams * (n_main + n_reserve);
    if (gid >= total_players || teams[gid] != loser_team) return;

    float sigma = 1.0f - 0.9f * iteration / max_iter;
    for (int j = 0; j < dim; j++) {
        float pos = positions[gid * dim + j];
        pos += gaussian_random(&seeds[gid], 0.0f, sigma);
        pos = max(bounds[j * 2], min(bounds[j * 2 + 1], pos));
        positions[gid * dim + j] = pos;
    }
}
