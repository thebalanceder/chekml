#ifndef SOA_H
#define SOA_H

#include "generaloptimizer.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define SOA_MU_MAX 0.9
#define SOA_MU_MIN 0.05
#define W_MAX_SOA 0.9
#define W_MIN_SOA 0.2
#define NUM_REGIONS 3

typedef struct {
    uint64_t magic; // Magic number to identify SOA structure
    Optimizer *opt;
    int num_regions;
    double mu_max, mu_min;
    double w_max, w_min;
    int *start_reg, *end_reg, *size_reg;
    double *rmax, *rmin;
    double *pbest_s, *pbest_fun;
    double *lbest_s, *lbest_fun;
    double *e_t_1, *e_t_2;
    double *f_t_1, *f_t_2;
} SOA;

SOA* SOA_init(Optimizer *opt);
void SOA_free(SOA *soa);
void SOA_optimize(void *soa_ptr, ObjectiveFunction objective_function);

#endif
