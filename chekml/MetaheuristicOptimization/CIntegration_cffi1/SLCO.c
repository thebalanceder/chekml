#include "SLCO.h"
#include <string.h>
#include <time.h>

// Comparison function for sorting players by cost
static int compare_players(const void *a, const void *b) {
    return (*(double *)a > *(double *)b) - (*(double *)a < *(double *)b);
}

// Initialize the league with random player positions
void initialize_league(SLCO_Optimizer *opt) {
    League *league = &opt->league;
    league->n_teams = SLCO_N_TEAMS;
    league->n_main_players = SLCO_N_MAIN_PLAYERS;
    league->n_reserve_players = SLCO_N_RESERVE_PLAYERS;
    league->best_team_idx = 0;
    league->best_total_cost = INFINITY;

    int total_players = league->n_teams * SLCO_TOTAL_PLAYERS_PER_TEAM;
    int dim = opt->base.dim;

    // Preallocate contiguous memory
    opt->all_positions = (double *)malloc(total_players * dim * sizeof(double));
    opt->all_costs = (double *)malloc(total_players * sizeof(double));
    opt->temp_buffer = (double *)malloc(dim * sizeof(double));
    if (!opt->all_positions || !opt->all_costs || !opt->temp_buffer) {
        fprintf(stderr, "Memory allocation failed for buffers\n");
        exit(1);
    }

    league->teams = (Team *)malloc(league->n_teams * sizeof(Team));
    if (!league->teams) {
        fprintf(stderr, "Memory allocation failed for teams\n");
        exit(1);
    }

    // Initialize RNG
    opt->rng_state = (uint64_t)time(NULL) ^ 0xDEADBEEF;

    // Assign memory blocks and initialize positions
    double *pos_ptr = opt->all_positions;
    double *cost_ptr = opt->all_costs;
    for (int i = 0; i < league->n_teams; i++) {
        league->teams[i].positions = pos_ptr;
        league->teams[i].costs = cost_ptr;
        league->teams[i].total_cost = 0.0;

        for (int j = 0; j < SLCO_TOTAL_PLAYERS_PER_TEAM; j++) {
            for (int k = 0; k < dim; k++) {
                double r = slco_fast_rand(&opt->rng_state);
                pos_ptr[k] = opt->base.bounds[2 * k] + 
                             r * (opt->base.bounds[2 * k + 1] - opt->base.bounds[2 * k]);
            }
            *cost_ptr = opt->objective_function(pos_ptr);
            pos_ptr += dim;
            cost_ptr++;
        }
    }

    takhsis(opt);
    update_total_cost(opt);
}

// Reassign players to teams based on sorted costs
void takhsis(SLCO_Optimizer *opt) {
    League *league = &opt->league;
    int total_players = league->n_teams * SLCO_TOTAL_PLAYERS_PER_TEAM;
    int dim = opt->base.dim;

    // Create index array for sorting
    int *indices = (int *)malloc(total_players * sizeof(int));
    if (!indices) {
        fprintf(stderr, "Memory allocation failed for indices\n");
        exit(1);
    }
    for (int i = 0; i < total_players; i++) {
        indices[i] = i;
    }

    // Sort indices by cost
    for (int i = 1; i < total_players; i++) {
        int j = i;
        while (j > 0 && opt->all_costs[indices[j]] < opt->all_costs[indices[j-1]]) {
            int temp = indices[j];
            indices[j] = indices[j-1];
            indices[j-1] = temp;
            j--;
        }
    }

    // Reassign positions and costs
    double *new_positions = (double *)malloc(total_players * dim * sizeof(double));
    double *new_costs = (double *)malloc(total_players * sizeof(double));
    if (!new_positions || !new_costs) {
        fprintf(stderr, "Memory allocation failed for new buffers\n");
        exit(1);
    }

    for (int i = 0; i < total_players; i++) {
        memcpy(new_positions + i * dim, opt->all_positions + indices[i] * dim, dim * sizeof(double));
        new_costs[i] = opt->all_costs[indices[i]];
    }

    memcpy(opt->all_positions, new_positions, total_players * dim * sizeof(double));
    memcpy(opt->all_costs, new_costs, total_players * sizeof(double));

    // Update team pointers
    double *pos_ptr = opt->all_positions;
    double *cost_ptr = opt->all_costs;
    for (int i = 0; i < league->n_teams; i++) {
        league->teams[i].positions = pos_ptr;
        league->teams[i].costs = cost_ptr;
        pos_ptr += SLCO_TOTAL_PLAYERS_PER_TEAM * dim;
        cost_ptr += SLCO_TOTAL_PLAYERS_PER_TEAM;
    }

    free(new_positions);
    free(new_costs);
    free(indices);
}

// Update total cost for each team
void update_total_cost(SLCO_Optimizer *opt) {
    League *league = &opt->league;
    for (int i = 0; i < league->n_teams; i++) {
        double mean_cost = 0.0;
        double *costs = league->teams[i].costs;
        for (int j = 0; j < SLCO_N_MAIN_PLAYERS; j++) {
            mean_cost += costs[j];
        }
        mean_cost /= SLCO_N_MAIN_PLAYERS;
        league->teams[i].total_cost = (mean_cost != 0.0) ? 1.0 / mean_cost : INFINITY;
        if (league->teams[i].total_cost < league->best_total_cost) {
            league->best_total_cost = league->teams[i].total_cost;
            league->best_team_idx = i;
        }
    }
}

// Determine winner and loser based on total cost probability
void probability_host(SLCO_Optimizer *opt, int ii, int jj, int *winner, int *loser) {
    League *league = &opt->league;
    double total_cost_ii = league->teams[ii].total_cost;
    double total_cost_jj = league->teams[jj].total_cost;
    double p_simple = total_cost_ii / (total_cost_ii + total_cost_jj);
    double r = slco_fast_rand(&opt->rng_state);
    if (r < p_simple) {
        *winner = ii;
        *loser = jj;
    } else {
        *winner = jj;
        *loser = ii;
    }
}

// Update main players of the winner team (SIMD-optimized)
void winner_function_main(SLCO_Optimizer *opt) {
    League *league = &opt->league;
    int winner = league->best_team_idx;
    int dim = opt->base.dim;
    double *xxx_pos = opt->temp_buffer;
    double *main_pos = league->teams[winner].positions;
    double *best_team_pos = league->teams[0].positions;

    for (int i = 0; i < SLCO_N_MAIN_PLAYERS; i++) {
        double w = 0.7 + 0.3 * slco_fast_rand(&opt->rng_state);
        double *curr_pos = main_pos + i * dim;

        #ifdef __AVX__
        for (int j = 0; j < dim; j += 4) {
            if (j + 4 > dim) break;
            __m256d curr = _mm256_loadu_pd(curr_pos + j);
            __m256d best = _mm256_loadu_pd(main_pos + j);
            __m256d global_best = _mm256_loadu_pd(best_team_pos + j);
            __m256d r1 = _mm256_set1_pd(slco_fast_rand(&opt->rng_state));
            __m256d r2 = _mm256_set1_pd(slco_fast_rand(&opt->rng_state));
            __m256d w_vec = _mm256_set1_pd(w);
            __m256d alpha = _mm256_set1_pd(SLCO_ALPHA);
            __m256d beta = _mm256_set1_pd(SLCO_BETA);

            __m256d diff1 = _mm256_sub_pd(best, curr);
            __m256d diff2 = _mm256_sub_pd(global_best, curr);
            __m256d term1 = _mm256_mul_pd(w_vec, curr);
            __m256d term2 = _mm256_mul_pd(alpha, _mm256_mul_pd(r1, diff1));
            __m256d term3 = _mm256_mul_pd(beta, _mm256_mul_pd(r2, diff2));
            __m256d result = _mm256_add_pd(term1, _mm256_add_pd(term2, term3));

            __m256d bounds_min = _mm256_loadu_pd(opt->base.bounds + 2 * j);
            __m256d bounds_max = _mm256_loadu_pd(opt->base.bounds + 2 * j + 4);
            result = _mm256_max_pd(result, bounds_min);
            result = _mm256_min_pd(result, bounds_max);

            _mm256_storeu_pd(xxx_pos + j, result);
        }
        for (int j = (dim / 4) * 4; j < dim; j++) {
            xxx_pos[j] = w * curr_pos[j] +
                         SLCO_ALPHA * slco_fast_rand(&opt->rng_state) * (main_pos[j] - curr_pos[j]) +
                         SLCO_BETA * slco_fast_rand(&opt->rng_state) * (best_team_pos[j] - curr_pos[j]);
            xxx_pos[j] = fmin(fmax(xxx_pos[j], opt->base.bounds[2 * j]), opt->base.bounds[2 * j + 1]);
        }
        #else
        for (int j = 0; j < dim; j++) {
            xxx_pos[j] = w * curr_pos[j] +
                         SLCO_ALPHA * slco_fast_rand(&opt->rng_state) * (main_pos[j] - curr_pos[j]) +
                         SLCO_BETA * slco_fast_rand(&opt->rng_state) * (best_team_pos[j] - curr_pos[j]);
            xxx_pos[j] = fmin(fmax(xxx_pos[j], opt->base.bounds[2 * j]), opt->base.bounds[2 * j + 1]);
        }
        #endif

        double cost = opt->objective_function(xxx_pos);
        if (cost < league->teams[winner].costs[i]) {
            memcpy(curr_pos, xxx_pos, dim * sizeof(double));
            league->teams[winner].costs[i] = cost;

            // Insertion sort for maintaining order
            for (int j = i; j > 0 && league->teams[winner].costs[j] < league->teams[winner].costs[j-1]; j--) {
                double temp_cost = league->teams[winner].costs[j];
                league->teams[winner].costs[j] = league->teams[winner].costs[j-1];
                league->teams[winner].costs[j-1] = temp_cost;
                double *pos_j = main_pos + j * dim;
                double *pos_jm1 = main_pos + (j-1) * dim;
                for (int k = 0; k < dim; k++) {
                    double temp = pos_j[k];
                    pos_j[k] = pos_jm1[k];
                    pos_jm1[k] = temp;
                }
            }
        }
    }
}

// Update reserve players of the winner team
void winner_function_reserve(SLCO_Optimizer *opt) {
    League *league = &opt->league;
    int winner = league->best_team_idx;
    int dim = opt->base.dim;
    double *rq = opt->temp_buffer;
    double *reserve_pos = league->teams[winner].positions + SLCO_N_MAIN_PLAYERS * dim;
    double *main_pos = league->teams[winner].positions;

    // Compute gravity
    double *gravity = (double *)malloc(dim * sizeof(double));
    if (!gravity) {
        fprintf(stderr, "Memory allocation failed in winner_function_reserve\n");
        exit(1);
    }
    for (int j = 0; j < dim; j++) {
        gravity[j] = 0.0;
        for (int i = 0; i < SLCO_N_MAIN_PLAYERS; i++) {
            gravity[j] += main_pos[i * dim + j];
        }
        gravity[j] /= SLCO_N_MAIN_PLAYERS;
    }

    double beta = 0.9 + 0.1 * slco_fast_rand(&opt->rng_state);
    int last_idx = SLCO_N_RESERVE_PLAYERS - 1;
    double *last_pos = reserve_pos + last_idx * dim;

    for (int j = 0; j < dim; j++) {
        rq[j] = gravity[j] + beta * (gravity[j] - last_pos[j]);
        rq[j] = fmin(fmax(rq[j], opt->base.bounds[2 * j]), opt->base.bounds[2 * j + 1]);
    }
    double cost = opt->objective_function(rq);

    if (cost < league->teams[winner].costs[SLCO_N_MAIN_PLAYERS + last_idx]) {
        memcpy(last_pos, rq, dim * sizeof(double));
        league->teams[winner].costs[SLCO_N_MAIN_PLAYERS + last_idx] = cost;
    }

    // Sort all players
    int total_players = SLCO_TOTAL_PLAYERS_PER_TEAM;
    double *costs = league->teams[winner].costs;
    double *positions = league->teams[winner].positions;

    for (int i = 1; i < total_players; i++) {
        int j = i;
        while (j > 0 && costs[j] < costs[j-1]) {
            double temp_cost = costs[j];
            costs[j] = costs[j-1];
            costs[j-1] = temp_cost;
            double *pos_j = positions + j * dim;
            double *pos_jm1 = positions + (j-1) * dim;
            for (int k = 0; k < dim; k++) {
                double temp = pos_j[k];
                pos_j[k] = pos_jm1[k];
                pos_jm1[k] = temp;
            }
            j--;
        }
    }

    free(gravity);
}

// Update main and reserve players of the loser team
void loser_function(SLCO_Optimizer *opt) {
    League *league = &opt->league;
    int loser = (league->best_team_idx + 1) % league->n_teams;
    int dim = opt->base.dim;
    double *new_pos = opt->temp_buffer;
    double *main_pos = league->teams[loser].positions;
    double *reserve_pos = main_pos + SLCO_N_MAIN_PLAYERS * dim;

    // Mutate three random main players
    for (int i = 0; i < 3; i++) {
        int idx = slco_fast_rand(&opt->rng_state) * SLCO_N_MAIN_PLAYERS;
        int n_mu = (int)(SLCO_MUTATION_PROB * dim + 0.5);
        double *curr_pos = main_pos + idx * dim;

        memcpy(new_pos, curr_pos, dim * sizeof(double));
        double sigma = 0.1 * (opt->base.bounds[1] - opt->base.bounds[0]);
        for (int k = 0; k < n_mu; k++) {
            int j = slco_fast_rand(&opt->rng_state) * dim;
            new_pos[j] += sigma * (slco_fast_rand(&opt->rng_state) - 0.5);
            new_pos[j] = fmin(fmax(new_pos[j], opt->base.bounds[2 * j]), opt->base.bounds[2 * j + 1]);
        }

        double new_cost = opt->objective_function(new_pos);
        if (new_cost < league->teams[loser].costs[idx]) {
            memcpy(curr_pos, new_pos, dim * sizeof(double));
            league->teams[loser].costs[idx] = new_cost;
        }
    }

    // Simplified crossover for reserve players
    int r1 = slco_fast_rand(&opt->rng_state) * SLCO_N_RESERVE_PLAYERS;
    int r2 = (r1 + 1) % SLCO_N_RESERVE_PLAYERS;
    double *x1 = reserve_pos + r1 * dim;
    double *x2 = reserve_pos + r2 * dim;

    for (int j = 0; j < dim; j++) {
        double alpha = slco_fast_rand(&opt->rng_state);
        new_pos[j] = alpha * x1[j] + (1.0 - alpha) * x2[j];
        new_pos[j] = fmin(fmax(new_pos[j], opt->base.bounds[2 * j]), opt->base.bounds[2 * j + 1]);
    }
    double new_cost = opt->objective_function(new_pos);

    if (new_cost < league->teams[loser].costs[SLCO_N_MAIN_PLAYERS + r1]) {
        memcpy(x1, new_pos, dim * sizeof(double));
        league->teams[loser].costs[SLCO_N_MAIN_PLAYERS + r1] = new_cost;
    }

    // Sort all players
    int total_players = SLCO_TOTAL_PLAYERS_PER_TEAM;
    double *costs = league->teams[loser].costs;
    double *positions = league->teams[loser].positions;

    for (int i = 1; i < total_players; i++) {
        int j = i;
        while (j > 0 && costs[j] < costs[j-1]) {
            double temp_cost = costs[j];
            costs[j] = costs[j-1];
            costs[j-1] = temp_cost;
            double *pos_j = positions + j * dim;
            double *pos_jm1 = positions + (j-1) * dim;
            for (int k = 0; k < dim; k++) {
                double temp = pos_j[k];
                pos_j[k] = pos_jm1[k];
                pos_jm1[k] = temp;
            }
            j--;
        }
    }
}

// Simulate competition between teams
void competition(SLCO_Optimizer *opt, int iteration) {
    League *league = &opt->league;
    for (int ii = 0; ii < league->n_teams - 1; ii++) {
        for (int jj = ii + 1; jj < league->n_teams; jj++) {
            int winner, loser;
            probability_host(opt, ii, jj, &winner, &loser);
            league->best_team_idx = winner;
            winner_function_main(opt);
            winner_function_reserve(opt);
            league->best_team_idx = loser;
            loser_function(opt);
            update_total_cost(opt);
        }
    }
}

// Free league memory
void free_league(SLCO_Optimizer *opt) {
    League *league = &opt->league;
    free(league->teams);
    free(opt->all_positions);
    free(opt->all_costs);
    free(opt->temp_buffer);
}

// Main optimization function
void SLCO_optimize(Optimizer *base_opt, ObjectiveFunction objective_function) {
    SLCO_Optimizer *opt = (SLCO_Optimizer *)malloc(sizeof(SLCO_Optimizer));
    if (!opt) {
        fprintf(stderr, "Memory allocation failed for SLCO_Optimizer\n");
        exit(1);
    }
    opt->base = *base_opt;
    opt->objective_function = objective_function;
    opt->all_positions = NULL;
    opt->all_costs = NULL;
    opt->temp_buffer = NULL;

    initialize_league(opt);

    for (int iter = 0; iter < opt->base.max_iter; iter++) {
        competition(opt, iter);

        double *best_pos = opt->league.teams[opt->league.best_team_idx].positions;
        double best_cost = opt->league.teams[opt->league.best_team_idx].costs[0];
        if (best_cost < opt->base.best_solution.fitness) {
            opt->base.best_solution.fitness = best_cost;
            memcpy(opt->base.best_solution.position, best_pos, opt->base.dim * sizeof(double));
        }
        enforce_bound_constraints(&opt->base);
    }

    *base_opt = opt->base;
    free_league(opt);
    free(opt);
}
