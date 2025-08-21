#define CRO_INITIAL_KE 1000.0f
#define CRO_MAX_POPULATION 1000

// Random number generator (XOR-shift)
inline float rand_float(uint *seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return (*seed) / 4294967296.0f;
}

// Initialize population
__kernel void initialize_population(
    __global float *position,
    __global float *pe,
    __global float *ke,
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
    __global float *pos = position + gid * dim;
    for (int j = 0; j < dim; j++) {
        float lower = bounds[j * 2];
        float upper = bounds[j * 2 + 1];
        pos[j] = lower + rand_float(&seeds[gid]) * (upper - lower);
    }

    // Initialize kinetic energy
    ke[gid] = CRO_INITIAL_KE;

    // Placeholder potential energy (to be evaluated on CPU)
    pe[gid] = INFINITY;
}

// On-wall ineffective collision
__kernel void on_wall_collision(
    __global float *position,
    __global float *pe,
    __global float *ke,
    __global float *bounds,
    __global uint *seeds,
    float mole_coll,
    float alpha,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    if (rand_float(&seeds[gid]) >= mole_coll) {
        __global float *pos = position + gid * dim;
        float old_pe = pe[gid];
        float old_ke = ke[gid];

        // Create new solution
        float new_pos[32]; // Assuming dim <= 32
        float r = rand_float(&seeds[gid]);
        for (int j = 0; j < dim; j++) {
            new_pos[j] = pos[j] + r * (bounds[j * 2 + 1] - bounds[j * 2]) * (rand_float(&seeds[gid]) * 2.0f - 1.0f);
            new_pos[j] = clamp(new_pos[j], bounds[j * 2], bounds[j * 2 + 1]);
        }

        // Placeholder for new_pe (to be evaluated on CPU)
        pe[gid] = INFINITY;
        for (int j = 0; j < dim; j++) {
            pos[j] = new_pos[j];
        }
    }
}

// Decomposition reaction
__kernel void decomposition(
    __global float *position,
    __global float *pe,
    __global float *ke,
    __global float *bounds,
    __global uint *seeds,
    float mole_coll,
    float alpha,
    float split_ratio,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size || pop_size >= CRO_MAX_POPULATION) return;

    if (rand_float(&seeds[gid]) >= mole_coll && rand_float(&seeds[gid]) >= 0.5f) {
        __global float *pos = position + gid * dim;
        float old_pe = pe[gid];
        float old_ke = ke[gid];
        if (old_pe + old_ke < alpha) return;

        // Create two new solutions
        float split1[32], split2[32]; // Assuming dim <= 32
        for (int j = 0; j < dim; j++) {
            split1[j] = pos[j] + split_ratio * (rand_float(&seeds[gid]) * 2.0f - 1.0f);
            split2[j] = pos[j] - split_ratio * (rand_float(&seeds[gid]) * 2.0f - 1.0f);
            split1[j] = clamp(split1[j], bounds[j * 2], bounds[j * 2 + 1]);
            split2[j] = clamp(split2[j], bounds[j * 2], bounds[j * 2 + 1]);
        }

        // Update current molecule
        for (int j = 0; j < dim; j++) {
            pos[j] = split1[j];
        }
        pe[gid] = INFINITY; // Placeholder
        ke[gid] = old_ke / 2.0f;

        // Add new molecule (simplified, assuming CPU handles population growth)
        if (gid + pop_size < CRO_MAX_POPULATION) {
            __global float *new_pos = position + (pop_size + gid) * dim;
            for (int j = 0; j < dim; j++) {
                new_pos[j] = split2[j];
            }
            pe[pop_size + gid] = INFINITY;
            ke[pop_size + gid] = old_ke / 2.0f;
        }
    }
}

// Inter-molecular ineffective collision
__kernel void inter_molecular_collision(
    __global float *position,
    __global float *pe,
    __global float *ke,
    __global float *bounds,
    __global uint *seeds,
    float mole_coll,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size - 1 || rand_float(&seeds[gid]) >= mole_coll) return;

    __global float *pos1 = position + gid * dim;
    __global float *pos2 = position + (gid + 1) * dim;
    float old_pe1 = pe[gid];
    float old_pe2 = pe[gid + 1];
    float old_ke1 = ke[gid];
    float old_ke2 = ke[gid + 1];

    float r1 = rand_float(&seeds[gid]);
    float r2 = rand_float(&seeds[gid]);
    float new_pos1[32], new_pos2[32]; // Assuming dim <= 32
    for (int j = 0; j < dim; j++) {
        new_pos1[j] = pos1[j] + r1 * (rand_float(&seeds[gid]) * 2.0f - 1.0f);
        new_pos2[j] = pos2[j] + r2 * (rand_float(&seeds[gid]) * 2.0f - 1.0f);
        new_pos1[j] = clamp(new_pos1[j], bounds[j * 2], bounds[j * 2 + 1]);
        new_pos2[j] = clamp(new_pos2[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    for (int j = 0; j < dim; j++) {
        pos1[j] = new_pos1[j];
        pos2[j] = new_pos2[j];
    }
    pe[gid] = INFINITY;
    pe[gid + 1] = INFINITY;
    ke[gid] = (old_ke1 + old_ke2) * rand_float(&seeds[gid]);
    ke[gid + 1] = (old_ke1 + old_ke2) - ke[gid];
}

// Synthesis reaction
__kernel void synthesis(
    __global float *position,
    __global float *pe,
    __global float *ke,
    __global float *bounds,
    __global uint *seeds,
    float mole_coll,
    float beta,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size - 1 || rand_float(&seeds[gid]) >= mole_coll || rand_float(&seeds[gid]) >= 0.5f) return;

    __global float *pos1 = position + gid * dim;
    __global float *pos2 = position + (gid + 1) * dim;
    float old_pe1 = pe[gid];
    float old_pe2 = pe[gid + 1];
    float old_ke1 = ke[gid];
    float old_ke2 = ke[gid + 1];

    float new_pos[32]; // Assuming dim <= 32
    for (int j = 0; j < dim; j++) {
        new_pos[j] = (pos1[j] + pos2[j]) / 2.0f;
        new_pos[j] = clamp(new_pos[j], bounds[j * 2], bounds[j * 2 + 1]);
    }

    for (int j = 0; j < dim; j++) {
        pos1[j] = new_pos[j];
    }
    pe[gid] = INFINITY;
    ke[gid] = old_ke1 + old_ke2;
    // Mark second molecule for removal (handled on CPU)
    pe[gid + 1] = INFINITY;
    ke[gid + 1] = 0.0f;
}

// Elimination phase
__kernel void elimination_phase(
    __global float *position,
    __global float *pe,
    __global float *ke,
    __global float *bounds,
    __global uint *seeds,
    float elim_ratio,
    float initial_ke,
    int dim,
    int pop_size
) {
    int gid = get_global_id(0);
    if (gid >= pop_size) return;

    // Simplified elimination: reset worst molecules
    int worst_count = (int)(elim_ratio * pop_size);
    if (gid >= pop_size - worst_count) {
        for (int j = 0; j < dim; j++) {
            float lower = bounds[j * 2];
            float upper = bounds[j * 2 + 1];
            position[gid * dim + j] = lower + rand_float(&seeds[gid]) * (upper - lower);
        }
        ke[gid] = initial_ke;
        pe[gid] = INFINITY;
    }
}
