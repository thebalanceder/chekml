#define GSO_LUCIFERIN_INITIAL 5.0f
#define GSO_DECISION_RANGE_INITIAL 3.0f
#define GSO_LUCIFERIN_DECAY 0.4f
#define GSO_LUCIFERIN_ENHANCEMENT 0.6f
#define GSO_NEIGHBOR_THRESHOLD 0.08f
#define GSO_STEP_SIZE 0.6f
#define GSO_SENSOR_RANGE 10.0f
#define GSO_NEIGHBOR_COUNT 10

// Random number generator (XOR-shift)
inline float rand_float(uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}

// Convert objective function value to minimization form
inline float convert_to_min(float fcn) {
    return (fcn >= 0.0f) ? 1.0f / (1.0f + fcn) : 1.0f + fabs(fcn);
}

// Initialize population
__kernel void initialize_population(
    __global float *population,
    __global float *fitness,
    __global float *decision_range,
    __global float *bounds,
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

    // Initialize luciferin and decision range
    fitness[gid] = GSO_LUCIFERIN_INITIAL;
    decision_range[gid] = GSO_DECISION_RANGE_INITIAL;
}

// Update luciferin
__kernel void update_luciferin(
    __global float *fitness,
    __global float *objective_values,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    float obj_val = objective_values[gid];
    float fitness_val = convert_to_min(obj_val);
    fitness[gid] = (1.0f - GSO_LUCIFERIN_DECAY) * fitness[gid] +
                   GSO_LUCIFERIN_ENHANCEMENT * fitness_val;
}

// Compute distances and perform movement phase
__kernel void movement_phase(
    __global float *population,
    __global float *fitness,
    __global float *decision_range,
    __global float *distances,
    __global float *bounds,
    __global uint *seeds,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    __global float *pos = population + gid * dim;
    float current_luciferin = fitness[gid];
    float current_range = decision_range[gid];

    // Compute distances
    for (int j = 0; j < pop_size; j++) {
        __global float *other_pos = population + j * dim;
        float dist = 0.0f;
        for (int k = 0; k < dim; k++) {
            float diff = pos[k] - other_pos[k];
            dist += diff * diff;
        }
        distances[gid * pop_size + j] = sqrt(dist);
    }

    // Find neighbors
    int neighbors[GSO_NEIGHBOR_COUNT];
    float probs[GSO_NEIGHBOR_COUNT];
    int neighbor_count = 0;
    float prob_sum = 0.0f;

    for (int j = 0; j < pop_size; j++) {
        if (distances[gid * pop_size + j] < current_range && fitness[j] > current_luciferin) {
            if (neighbor_count < GSO_NEIGHBOR_COUNT) {
                neighbors[neighbor_count] = j;
                probs[neighbor_count] = fitness[j] - current_luciferin;
                prob_sum += probs[neighbor_count];
                neighbor_count++;
            }
        }
    }

    if (neighbor_count == 0) return;

    // Normalize probabilities
    if (prob_sum > 0.0f) {
        for (int k = 0; k < neighbor_count; k++) {
            probs[k] /= prob_sum;
        }
    } else {
        return;
    }

    // Select neighbor using roulette wheel
    float rn = rand_float(&seeds[gid]);
    float cum_prob = 0.0f;
    int selected_idx = neighbors[neighbor_count - 1];
    for (int k = 0; k < neighbor_count; k++) {
        cum_prob += probs[k];
        if (cum_prob >= rn) {
            selected_idx = neighbors[k];
            break;
        }
    }

    // Move towards selected neighbor
    __global float *selected_pos = population + selected_idx * dim;
    float distance = distances[gid * pop_size + selected_idx];
    if (distance > 0.0f) {
        for (int k = 0; k < dim; k++) {
            pos[k] += GSO_STEP_SIZE * (selected_pos[k] - pos[k]) / distance;
            pos[k] = clamp(pos[k], bounds[k * 2], bounds[k * 2 + 1]);
        }
    }
}

// Update decision range
__kernel void update_decision_range(
    __global float *decision_range,
    __global float *fitness,
    __global float *distances,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    float current_luciferin = fitness[gid];
    float current_range = decision_range[gid];
    int neighbor_count = 0;

    for (int j = 0; j < pop_size; j++) {
        if (distances[gid * pop_size + j] < current_range && fitness[j] > current_luciferin) {
            neighbor_count++;
        }
    }

    decision_range[gid] = min(GSO_SENSOR_RANGE,
                             max(0.0f, current_range +
                                       GSO_NEIGHBOR_THRESHOLD * (GSO_NEIGHBOR_COUNT - neighbor_count)));
}
