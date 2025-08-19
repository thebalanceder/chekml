#define IWD_A_S 1.0f
#define IWD_B_S 0.01f
#define IWD_C_S 1.0f
#define IWD_A_V 1.0f
#define IWD_B_V 0.01f
#define IWD_C_V 1.0f
#define IWD_INIT_VEL 200.0f
#define IWD_P_N 0.9f
#define IWD_P_IWD 0.9f
#define IWD_INITIAL_SOIL 10000.0f
#define IWD_EPSILON_S 0.0001f

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
    __global float *soil,
    __global float *hud,
    __global float *velocities,
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

    // Initialize soil matrix
    for (int j = 0; j < pop_size; j++) {
        soil[gid * pop_size + j] = IWD_INITIAL_SOIL;
    }

    // Initialize HUD matrix
    for (int j = 0; j < pop_size; j++) {
        if (j == gid) {
            hud[gid * pop_size + j] = 0.0f;
        } else {
            float dist = 0.0f;
            __global float *other_pos = population + j * dim;
            for (int k = 0; k < dim; k++) {
                float diff = pos[k] - other_pos[k];
                dist += diff * diff;
            }
            hud[gid * pop_size + j] = sqrt(dist);
        }
    }

    // Initialize velocity
    velocities[gid] = IWD_INIT_VEL;

    // Placeholder fitness
    fitness[gid] = INFINITY;
}

// Compute g(soil) for probability calculation
inline float g_soil(__global float *soil, int pop_size, int i, int j, __global char *visited_flags) {
    float minimum = INFINITY;
    for (int l = 0; l < pop_size; l++) {
        if (!visited_flags[l] && soil[i * pop_size + l] < minimum) {
            minimum = soil[i * pop_size + l];
        }
    }
    if (minimum >= 0.0f) {
        return soil[i * pop_size + j];
    }
    return soil[i * pop_size + j] - minimum;
}

// Compute f(soil) for probability calculation
inline float f_soil(__global float *soil, int pop_size, int i, int j, __global char *visited_flags) {
    return 1.0f / (IWD_EPSILON_S + g_soil(soil, pop_size, i, j, visited_flags));
}

// Move water drop
__kernel void move_water_drop(
    __global float *population,
    __global float *fitness,
    __global float *bounds,
    __global uint *seeds,
    __global float *soil,
    __global float *hud,
    __global float *velocities,
    __global char *visited_flags,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    float velocity = velocities[gid];
    int current = gid;

    // Compute probabilities for unvisited nodes
    float probabilities[32]; // Assuming pop_size <= 32
    int valid_nodes[32];
    int valid_count = 0;
    float sum_fsoil = 0.0f;

    for (int j = 0; j < pop_size; j++) {
        if (!visited_flags[j]) {
            float fsoil = f_soil(soil, pop_size, current, j, visited_flags);
            probabilities[valid_count] = fsoil;
            sum_fsoil += fsoil;
            valid_nodes[valid_count] = j;
            valid_count++;
        }
    }

    // Select next node
    int next_node = current;
    if (valid_count > 0 && sum_fsoil > 0.0f) {
        float random_number = rand_float(&seeds[gid]);
        float probability_sum = 0.0f;
        for (int v = 0; v < valid_count; v++) {
            probability_sum += probabilities[v] / sum_fsoil;
            if (random_number < probability_sum) {
                next_node = valid_nodes[v];
                break;
            }
        }
    }

    if (next_node != current) {
        // Update velocity
        velocity += IWD_A_V / (IWD_B_V + IWD_C_V * soil[current * pop_size + next_node] * 
                              soil[current * pop_size + next_node]);
        velocities[gid] = velocity;

        // Update soil
        float delta_soil = IWD_A_S / (IWD_B_S + IWD_C_S * hud[current * pop_size + next_node] / velocity * 
                                     hud[current * pop_size + next_node] / velocity);
        soil[current * pop_size + next_node] = (1.0f - IWD_P_N) * soil[current * pop_size + next_node] - 
                                              IWD_P_N * delta_soil;

        // Move water drop (interpolate position)
        __global float *other_pos = population + next_node * dim;
        for (int k = 0; k < dim; k++) {
            pos[k] = 0.5f * (pos[k] + other_pos[k]);
            pos[k] = clamp(pos[k], bounds[k * 2], bounds[k * 2 + 1]);
        }

        // Mark as visited
        visited_flags[next_node] = 1;
    }

    // Placeholder fitness
    fitness[gid] = INFINITY;
}

// Update soil for iteration best
__kernel void update_soil(
    __global float *soil,
    __global int *visited,
    int visited_count,
    float soil_amount,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= visited_count - 1) return;

    int prev = visited[gid];
    int curr = visited[gid + 1];
    soil[prev * pop_size + curr] = (1.0f + IWD_P_IWD) * soil[prev * pop_size + curr] -
                                  IWD_P_IWD * (1.0f / (pop_size - 1)) * soil_amount;
}
