/* EHO.h - Header file for Elephant Herding Optimization */
#ifndef EHO_H
#define EHO_H

#include "generaloptimizer.h"

// EHO Parameters
#define EHO_NUM_CLANS 5
#define EHO_ALPHA 0.5
#define EHO_BETA 0.1
#define EHO_KEEP 2

// Function Declarations
void eho_initialize_population(Optimizer *opt, ObjectiveFunction objective_function, double *temp_position);
void eho_clan_division_phase(Optimizer *opt, double *temp_position);
void eho_clan_updating_phase(Optimizer *opt);
void eho_separating_phase(Optimizer *opt);
void eho_elitism_phase(Optimizer *opt, Solution *elite, double *temp_position);
void eho_update_best_solution(Optimizer *opt, ObjectiveFunction objective_function);
void EHO_optimize(void *opt_void, ObjectiveFunction objective_function);

#endif
