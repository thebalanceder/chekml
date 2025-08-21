// Define constants used in the kernels
#define SGO_ATTACK_RATE 0.5f
#define SGO_DEFENSE_STRENGTH 0.3f
#define SGO_FIGHT_INTENSITY 0.2f
#define SGO_WIN_THRESHOLD 0.6f

inline float rand_float(__global uint *seed) {
    *seed = (*seed * 1103515245u + 12345u) & 0x7fffffffu;
    return (float)(*seed) / (float)0x7fffffffu;
}

__kernel void initialize_players(__global float *positions, __global float *bounds,
                                __global uint *seeds, int dim, int population_size) {
    int gid = get_global_id(0);
    if (gid >= population_size) return;

    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        positions[gid * dim + j] = lower + (upper - lower) * rand_float(&seeds[gid]);
    }
}

__kernel void divide_teams(__global int *team_assignments, __global uint *seeds,
                           int population_size, int offensive_size) {
    int gid = get_global_id(0);
    if (gid >= population_size) return;

    float r = rand_float(&seeds[gid]);
    team_assignments[gid] = (r < (float)(offensive_size) / population_size) ? 0 : 1; // 0: offensive, 1: defensive
}

__kernel void simulate_fight(__global float *positions, __global int *team_assignments,
                             __global float *bounds, __global uint *seeds,
                             int dim, int population_size, int offensive_size) {
    int gid = get_global_id(0);
    int defensive_size = population_size - offensive_size;
    if (gid >= min(offensive_size, defensive_size)) return;

    // Find offensive and defensive indices
    int off_idx = -1, def_idx = -1;
    int off_count = 0, def_count = 0;
    for (int i = 0; i < population_size; i++) {
        if (team_assignments[i] == 0 && off_count == gid) {
            off_idx = i;
            off_count++;
        } else if (team_assignments[i] == 0) {
            off_count++;
        }
        if (team_assignments[i] == 1 && def_count == gid) {
            def_idx = i;
            def_count++;
        } else if (team_assignments[i] == 1) {
            def_count++;
        }
        if (off_idx != -1 && def_idx != -1) break;
    }

    if (off_idx == -1 || def_idx == -1) return;

    float r1 = rand_float(&seeds[gid]);
    for (int j = 0; j < dim; j++) {
        float off_pos = positions[off_idx * dim + j];
        float def_pos = positions[def_idx * dim + j];
        float diff = off_pos - def_pos;
        float movement = SGO_FIGHT_INTENSITY * diff * r1;
        float resistance = SGO_DEFENSE_STRENGTH * diff * (1.0f - r1);

        float new_off_pos = off_pos + movement;
        float new_def_pos = def_pos + resistance;

        new_off_pos = max(bounds[j * 2], min(bounds[j * 2 + 1], new_off_pos));
        new_def_pos = max(bounds[j * 2], min(bounds[j * 2 + 1], new_def_pos));

        positions[off_idx * dim + j] = new_off_pos;
        positions[def_idx * dim + j] = new_def_pos;
    }
}

__kernel void determine_winners(__global float *costs, __global int *team_assignments,
                                __global int *winners, int population_size, int offensive_size) {
    int gid = get_global_id(0);
    int defensive_size = population_size - offensive_size;
    if (gid >= min(offensive_size, defensive_size)) return;

    // Find offensive and defensive indices
    int off_idx = -1, def_idx = -1;
    int off_count = 0, def_count = 0;
    for (int i = 0; i < population_size; i++) {
        if (team_assignments[i] == 0 && off_count == gid) {
            off_idx = i;
            off_count++;
        } else if (team_assignments[i] == 0) {
            off_count++;
        }
        if (team_assignments[i] == 1 && def_count == gid) {
            def_idx = i;
            def_count++;
        } else if (team_assignments[i] == 1) {
            def_count++;
        }
        if (off_idx != -1 && def_idx != -1) break;
    }

    if (off_idx == -1 || def_idx == -1) return;

    float off_fitness = costs[off_idx];
    float def_fitness = costs[def_idx];

    if (off_fitness < def_fitness * SGO_WIN_THRESHOLD) {
        winners[gid] = off_idx;
    } else if (def_fitness < off_fitness * SGO_WIN_THRESHOLD) {
        winners[gid] = def_idx;
    } else {
        winners[gid] = -1; // No winner
    }
}

__kernel void update_positions(__global float *positions, __global float *best_position,
                               __global float *bounds, __global uint *seeds,
                               __global int *winners, int dim, int population_size,
                               int winner_count) {
    int gid = get_global_id(0);
    if (gid >= population_size) return;

    int is_winner = 0;
    for (int w = 0; w < winner_count; w++) {
        if (winners[w] == gid) {
            is_winner = 1;
            break;
        }
    }

    if (is_winner) {
        float r2 = rand_float(&seeds[gid]);
        for (int j = 0; j < dim; j++) {
            float pos = positions[gid * dim + j];
            float best = best_position[j];
            positions[gid * dim + j] = pos + SGO_ATTACK_RATE * r2 * (best - pos);
        }
    } else {
        for (int j = 0; j < dim; j++) {
            float r3 = rand_float(&seeds[gid]);
            float lower = bounds[j * 2];
            float upper = bounds[j * 2 + 1];
            positions[gid * dim + j] += SGO_FIGHT_INTENSITY * r3 * (upper - lower);
        }
    }

    // Bounds checking
    for (int j = 0; j < dim; j++) {
        float pos = positions[gid * dim + j];
        positions[gid * dim + j] = max(bounds[j * 2], min(bounds[j * 2 + 1], pos));
    }
}
