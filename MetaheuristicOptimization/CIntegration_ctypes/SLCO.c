#include "SLCO.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

// Comparison function for sorting players by cost
static int compare_players(const void *a, const void *b) {
    double cost_a = ((Player *)a)->cost;
    double cost_b = ((Player *)b)->cost;
    return (cost_a > cost_b) - (cost_a < cost_b);
}

// Generate Gaussian random number (Box-Muller transform)
static double gaussian_random(double mean, double stddev) {
    static int has_spare = 0;
    static double spare;
    if (has_spare) {
        has_spare = 0;
        return mean + stddev * spare;
    }
    has_spare = 1;
    double u, v, s;
    do {
        u = 2.0 * rand() / RAND_MAX - 1.0;
        v = 2.0 * rand() / RAND_MAX - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + stddev * u * s;
}

// Initialize the league with random player positions
void initialize_league(SLCO_Optimizer *opt) {
    League *league = &opt->league;
    league->n_teams = SLCO_N_TEAMS;
    league->n_main_players = SLCO_N_MAIN_PLAYERS;
    league->n_reserve_players = SLCO_N_RESERVE_PLAYERS;
    league->best_team_idx = 0;
    league->best_total_cost = INFINITY;
    opt->stagnation_count = 0;
    opt->prev_best_cost = INFINITY;

    // Preallocate temporary buffers
    int total_players = league->n_teams * (league->n_main_players + league->n_reserve_players);
    opt->all_players = (Player *)calloc(total_players, sizeof(Player));
    opt->temp_buffer = (double *)calloc(opt->base.dim, sizeof(double));
    if (!opt->all_players || !opt->temp_buffer) {
        fprintf(stderr, "Memory allocation failed for buffers\n");
        exit(1);
    }

    league->teams = (Team *)calloc(league->n_teams, sizeof(Team));
    if (!league->teams) {
        fprintf(stderr, "Memory allocation failed for teams\n");
        exit(1);
    }

    for (int i = 0; i < league->n_teams; i++) {
        league->teams[i].main_players = (Player *)calloc(league->n_main_players, sizeof(Player));
        league->teams[i].reserve_players = (Player *)calloc(league->n_reserve_players, sizeof(Player));
        if (!league->teams[i].main_players || !league->teams[i].reserve_players) {
            fprintf(stderr, "Memory allocation failed for players\n");
            exit(1);
        }

        for (int j = 0; j < league->n_main_players; j++) {
            league->teams[i].main_players[j].position = (double *)calloc(opt->base.dim, sizeof(double));
            if (!league->teams[i].main_players[j].position) {
                fprintf(stderr, "Memory allocation failed for position\n");
                exit(1);
            }
            for (int k = 0; k < opt->base.dim; k++) {
                double r = (double)rand() / RAND_MAX;
                league->teams[i].main_players[j].position[k] = opt->base.bounds[2 * k] + 
                    r * (opt->base.bounds[2 * k + 1] - opt->base.bounds[2 * k]);
            }
            league->teams[i].main_players[j].cost = opt->objective_function(league->teams[i].main_players[j].position);
        }

        for (int j = 0; j < league->n_reserve_players; j++) {
            league->teams[i].reserve_players[j].position = (double *)calloc(opt->base.dim, sizeof(double));
            if (!league->teams[i].reserve_players[j].position) {
                fprintf(stderr, "Memory allocation failed for position\n");
                exit(1);
            }
            for (int k = 0; k < opt->base.dim; k++) {
                double r = (double)rand() / RAND_MAX;
                league->teams[i].reserve_players[j].position[k] = opt->base.bounds[2 * k] + 
                    r * (opt->base.bounds[2 * k + 1] - opt->base.bounds[2 * k]);
            }
            league->teams[i].reserve_players[j].cost = opt->objective_function(league->teams[i].reserve_players[j].position);
        }
        league->teams[i].total_cost = 0.0;
    }

    takhsis(opt);
    update_total_cost(opt);
}

// Reassign players to teams based on sorted costs
void takhsis(SLCO_Optimizer *opt) {
    League *league = &opt->league;
    int total_players = league->n_teams * (league->n_main_players + league->n_reserve_players);
    
    // Gather all players
    int idx = 0;
    for (int i = 0; i < league->n_teams; i++) {
        for (int j = 0; j < league->n_main_players; j++) {
            opt->all_players[idx++] = league->teams[i].main_players[j];
        }
        for (int j = 0; j < league->n_reserve_players; j++) {
            opt->all_players[idx++] = league->teams[i].reserve_players[j];
        }
    }

    // Sort all players by cost
    qsort(opt->all_players, total_players, sizeof(Player), compare_players);

    // Reassign players to teams
    idx = 0;
    for (int i = 0; i < league->n_teams; i++) {
        for (int j = 0; j < league->n_main_players; j++) {
            league->teams[i].main_players[j] = opt->all_players[idx++];
        }
        for (int j = 0; j < league->n_reserve_players; j++) {
            league->teams[i].reserve_players[j] = opt->all_players[idx++];
        }
        league->teams[i].total_cost = 0.0;
    }
}

// Update total cost for each team
void update_total_cost(SLCO_Optimizer *opt) {
    League *league = &opt->league;
    for (int i = 0; i < league->n_teams; i++) {
        double mean_cost = 0.0;
        for (int j = 0; j < league->n_main_players; j++) {
            mean_cost += league->teams[i].main_players[j].cost;
        }
        mean_cost /= league->n_main_players;
        league->teams[i].total_cost = (mean_cost != 0.0) ? 1.0 / mean_cost : INFINITY;
        if (league->teams[i].total_cost < league->best_total_cost) {
            league->best_total_cost = league->teams[i].total_cost;
            league->best_team_idx = i;
        }
    }
}

// Determine winner and loser based on total cost probability
void probability_host(League *league, int ii, int jj, int *winner, int *loser) {
    double total_cost_ii = league->teams[ii].total_cost;
    double total_cost_jj = league->teams[jj].total_cost;
    double p_simple = total_cost_ii / (total_cost_ii + total_cost_jj);
    double r = (double)rand() / RAND_MAX;
    if (r < p_simple) {
        *winner = ii;
        *loser = jj;
    } else {
        *winner = jj;
        *loser = ii;
    }
}

// Update main players of the winner team
void winner_function_main(SLCO_Optimizer *opt, int iteration) {
    League *league = &opt->league;
    int winner = league->best_team_idx;
    double *xxx_pos = opt->temp_buffer;
    double alpha = SLCO_ALPHA_MAX - (SLCO_ALPHA_MAX - SLCO_ALPHA_MIN) * iteration / opt->base.max_iter;
    double beta = SLCO_BETA_MAX - (SLCO_BETA_MAX - SLCO_BETA_MIN) * iteration / opt->base.max_iter;

    for (int i = 0; i < league->n_main_players; i++) {
        double w = 0.7 + 0.3 * ((double)rand() / RAND_MAX);
        for (int j = 0; j < opt->base.dim; j++) {
            // Differential evolution-inspired update
            int r1 = rand() % league->n_main_players;
            int r2 = rand() % league->n_main_players;
            while (r1 == r2 || r1 == i || r2 == i) {
                r1 = rand() % league->n_main_players;
                r2 = rand() % league->n_main_players;
            }
            xxx_pos[j] = league->teams[winner].main_players[i].position[j] +
                         alpha * (league->teams[winner].main_players[r1].position[j] -
                                  league->teams[winner].main_players[r2].position[j]) +
                         beta * (league->teams[0].main_players[0].position[j] -
                                 league->teams[winner].main_players[i].position[j]);
            xxx_pos[j] = fmin(fmax(xxx_pos[j], opt->base.bounds[2 * j]), opt->base.bounds[2 * j + 1]);
        }
        double cost = opt->objective_function(xxx_pos);

        if (cost < league->teams[winner].main_players[i].cost) {
            for (int j = 0; j < opt->base.dim; j++) {
                league->teams[winner].main_players[i].position[j] = xxx_pos[j];
            }
            league->teams[winner].main_players[i].cost = cost;

            // Insert player in sorted order
            for (int j = i; j > 0 && league->teams[winner].main_players[j].cost < 
                 league->teams[winner].main_players[j-1].cost; j--) {
                Player temp = league->teams[winner].main_players[j];
                league->teams[winner].main_players[j] = league->teams[winner].main_players[j-1];
                league->teams[winner].main_players[j-1] = temp;
            }
        }
    }
}

// Update reserve players of the winner team
void winner_function_reserve(SLCO_Optimizer *opt, int iteration) {
    League *league = &opt->league;
    int winner = league->best_team_idx;
    double *rq = opt->temp_buffer;
    double beta = SLCO_BETA_MAX - (SLCO_BETA_MAX - SLCO_BETA_MIN) * iteration / opt->base.max_iter;

    double *gravity = (double *)calloc(opt->base.dim, sizeof(double));
    if (!gravity) {
        fprintf(stderr, "Memory allocation failed in winner_function_reserve\n");
        exit(1);
    }

    // Calculate center of gravity for main players
    for (int j = 0; j < opt->base.dim; j++) {
        gravity[j] = 0.0;
        for (int jj = 0; jj < league->n_main_players; jj++) {
            gravity[j] += league->teams[winner].main_players[jj].position[j];
        }
        gravity[j] /= league->n_main_players;
    }

    // Update all reserve players
    for (int i = 0; i < league->n_reserve_players; i++) {
        for (int j = 0; j < opt->base.dim; j++) {
            double x = league->teams[winner].reserve_players[i].position[j];
            rq[j] = gravity[j] + beta * (gravity[j] - x);
            rq[j] = fmin(fmax(rq[j], opt->base.bounds[2 * j]), opt->base.bounds[2 * j + 1]);
        }
        double cost = opt->objective_function(rq);

        if (cost < league->teams[winner].reserve_players[i].cost) {
            for (int j = 0; j < opt->base.dim; j++) {
                league->teams[winner].reserve_players[i].position[j] = rq[j];
            }
            league->teams[winner].reserve_players[i].cost = cost;
        }
    }

    // Sort all players for reassignment
    int total_players = league->n_main_players + league->n_reserve_players;
    memcpy(opt->all_players, league->teams[winner].main_players, league->n_main_players * sizeof(Player));
    memcpy(opt->all_players + league->n_main_players, league->teams[winner].reserve_players, 
           league->n_reserve_players * sizeof(Player));
    qsort(opt->all_players, total_players, sizeof(Player), compare_players);

    memcpy(league->teams[winner].main_players, opt->all_players, league->n_main_players * sizeof(Player));
    memcpy(league->teams[winner].reserve_players, opt->all_players + league->n_main_players, 
           league->n_reserve_players * sizeof(Player));
    free(gravity);
}

// Update main and reserve players of the loser team
void loser_function(SLCO_Optimizer *opt, int iteration) {
    League *league = &opt->league;
    int loser = (league->best_team_idx + 1) % league->n_teams;
    double *new_pos = opt->temp_buffer;
    double alpha = SLCO_ALPHA_MAX - (SLCO_ALPHA_MAX - SLCO_ALPHA_MIN) * iteration / opt->base.max_iter;

    // Mutate main players
    int n_mutate = (int)(0.5 * league->n_main_players);
    for (int i = 0; i < n_mutate; i++) {
        int idx = rand() % league->n_main_players;
        int n_mu = (int)ceil(SLCO_MUTATION_PROB * opt->base.dim);
        int *r_indices = (int *)calloc(n_mu, sizeof(int));
        if (!r_indices) {
            fprintf(stderr, "Memory allocation failed for r_indices\n");
            exit(1);
        }
        for (int k = 0; k < n_mu; k++) {
            r_indices[k] = rand() % opt->base.dim;
        }

        double sigma = 0.1 * (opt->base.bounds[1] - opt->base.bounds[0]) * (1.0 - (double)iteration / opt->base.max_iter);
        for (int j = 0; j < opt->base.dim; j++) {
            new_pos[j] = league->teams[loser].main_players[idx].position[j];
        }
        for (int k = 0; k < n_mu; k++) {
            new_pos[r_indices[k]] += gaussian_random(0.0, sigma);
            new_pos[r_indices[k]] = fmin(fmax(new_pos[r_indices[k]], opt->base.bounds[2 * r_indices[k]]), 
                                         opt->base.bounds[2 * r_indices[k] + 1]);
        }
        double new_cost = opt->objective_function(new_pos);
        if (new_cost < league->teams[loser].main_players[idx].cost) {
            for (int j = 0; j < opt->base.dim; j++) {
                league->teams[loser].main_players[idx].position[j] = new_pos[j];
            }
            league->teams[loser].main_players[idx].cost = new_cost;
        }
        free(r_indices);
    }

    // Crossover for reserve players
    int r1 = rand() % league->n_reserve_players;
    int r2 = (r1 + 1) % league->n_reserve_players;
    double *x1 = league->teams[loser].reserve_players[r1].position;
    double *x2 = league->teams[loser].reserve_players[r2].position;

    for (int j = 0; j < opt->base.dim; j++) {
        new_pos[j] = alpha * x1[j] + (1.0 - alpha) * x2[j];
        new_pos[j] = fmin(fmax(new_pos[j], opt->base.bounds[2 * j]), opt->base.bounds[2 * j + 1]);
    }
    double new_cost = opt->objective_function(new_pos);

    if (new_cost < league->teams[loser].reserve_players[r1].cost) {
        for (int j = 0; j < opt->base.dim; j++) {
            league->teams[loser].reserve_players[r1].position[j] = new_pos[j];
        }
        league->teams[loser].reserve_players[r1].cost = new_cost;
    }

    // Sort all players
    int total_players = league->n_main_players + league->n_reserve_players;
    memcpy(opt->all_players, league->teams[loser].main_players, league->n_main_players * sizeof(Player));
    memcpy(opt->all_players + league->n_main_players, league->teams[loser].reserve_players, 
           league->n_reserve_players * sizeof(Player));
    qsort(opt->all_players, total_players, sizeof(Player), compare_players);

    memcpy(league->teams[loser].main_players, opt->all_players, league->n_main_players * sizeof(Player));
    memcpy(league->teams[loser].reserve_players, opt->all_players + league->n_main_players, 
           league->n_reserve_players * sizeof(Player));
}

// Simulate competition between teams
void competition(SLCO_Optimizer *opt, int iteration) {
    League *league = &opt->league;
    for (int ii = 0; ii < league->n_teams - 1; ii++) {
        for (int jj = ii + 1; jj < league->n_teams; jj++) {
            int winner, loser;
            probability_host(league, ii, jj, &winner, &loser);
            league->best_team_idx = winner;
            winner_function_main(opt, iteration);
            winner_function_reserve(opt, iteration);
            league->best_team_idx = loser;
            loser_function(opt, iteration);
            update_total_cost(opt);
        }
    }
}

// Free league memory
void free_league(SLCO_Optimizer *opt) {
    League *league = &opt->league;
    for (int i = 0; i < league->n_teams; i++) {
        for (int j = 0; j < league->n_main_players; j++) {
            free(league->teams[i].main_players[j].position);
        }
        for (int j = 0; j < league->n_reserve_players; j++) {
            free(league->teams[i].reserve_players[j].position);
        }
        free(league->teams[i].main_players);
        free(league->teams[i].reserve_players);
    }
    free(league->teams);
    free(opt->all_players);
    free(opt->temp_buffer);
}

// Main optimization function
void SLCO_optimize(Optimizer *base_opt, ObjectiveFunction objective_function) {
    static int seed_initialized = 0;
    if (!seed_initialized) {
        srand((unsigned)time(NULL));
        seed_initialized = 1;
    }

    SLCO_Optimizer *opt = (SLCO_Optimizer *)calloc(1, sizeof(SLCO_Optimizer));
    if (!opt) {
        fprintf(stderr, "Memory allocation failed for SLCO_Optimizer\n");
        exit(1);
    }
    opt->base = *base_opt;
    opt->objective_function = objective_function;
    opt->all_players = NULL;
    opt->temp_buffer = NULL;

    initialize_league(opt);

    for (int iter = 0; iter < opt->base.max_iter; iter++) {
        competition(opt, iter);

        Team *best_team = &opt->league.teams[opt->league.best_team_idx];
        Player *best_player = &best_team->main_players[0];
        if (best_player->cost < opt->base.best_solution.fitness) {
            opt->base.best_solution.fitness = best_player->cost;
            for (int j = 0; j < opt->base.dim; j++) {
                opt->base.best_solution.position[j] = best_player->position[j];
            }
            // Check for convergence
            if (fabs(opt->prev_best_cost - best_player->cost) < SLCO_CONVERGENCE_TOL) {
                opt->stagnation_count++;
            } else {
                opt->stagnation_count = 0;
            }
            opt->prev_best_cost = best_player->cost;
            if (opt->stagnation_count >= SLCO_STAGNATION_THRESHOLD) {
                break; // Early stopping
            }
        }
        enforce_bound_constraints(&opt->base);
    }

    *base_opt = opt->base;
    free_league(opt);
    free(opt);
}
