/* generaloptimizer.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>  // Added for time()
#include <omp.h>   // For parallelization
#include "generaloptimizer.h"
#include "DISOwithRCF.h"
#include "GPC.h"
#include "GWCA.h"
#include "SPO.h"
#include "ANS.h"
#include "BSA.h"
#include "CS.h"
#include "DE.h"
#include "EA.h"
#include "GA.h"
#include "HS.h"
#include "MA.h"
#include "SFS.h"
#include "GLS.h"
#include "ILS.h"
#include "SA.h"
#include "TS.h"
#include "VNS.h"
#include "AFSA.h"
#include "ES.h"
#include "HBO.h"
#include "KA.h"
#include "PO.h"
#include "SO.h"
#include "ADS.h"
#include "AAA.h"
#include "BCO.h"
#include "BBM.h"
#include "CFO.h"
#include "CBO.h"
#include "CRO.h"
#include "CA.h"
#include "CulA.h"
#include "FA.h"
#include "FFO.h"
#include "SDS.h"
#include "KCA.h"
#include "LS.h"
#include "WCA.h"
#include "ACO.h"
#include "ALO.h"
#include "ABC.h"
#include "BA.h"
#include "CSO.h"
#include "CroSA.h"
#include "CO.h"
#include "CucS.h"
#include "EHO.h"
#include "DEA.h"
#include "EPO.h"
#include "FHO.h"
#include "FirefA.h"
#include "FlyFO.h"
#include "HHO.h"
#include "GWO.h"
#include "JA.h"
#include "KHO.h"
#include "MfA.h"
#include "LOA.h"
#include "MSO.h"
#include "MFO.h"
#include "OSA.h"
#include "PuO.h"
#include "RDA.h"
#include "SFL.h"
#include "SMO.h"
#include "SWO.h"
#include "SSA.h"
#include "CMOA.h"
#include "WO.h"
#include "WOA.h"
#include "FSA.h"
#include "BBO.h"
#include "ICA.h"
#include "SOA.h"
#include "SEO.h"
#include "SLCO.h"
#include "TLBO.h"
#include "DRA.h"
#include "SGO.h"
#include "CheRO.h"
#include "CSS.h"
#include "BHA.h"
#include "EFO.h"
#include "EVO.h"
#include "LSA.h"
#include "SCA.h"
#include "SAO.h"
#include "TEO.h"
#include "TFWO.h"
#include "PVS.h"
#include "ARFO.h"
#include "FPA.h"
#include "POA.h"
#include "IWO.h"
#include "WPA.h"
#include "BSO.h"
#include "AMO.h"
#include "COA.h"
#include "GlowSO.h"
#include "GalSO.h"
#include "DHLO.h"
#include "HPO.h"
#include "IWD.h"
#include "JOA.h"
#include "WGMO.h"
#include "LoSA.h"
#include "RMO.h"
#include "PSO.h"
#include "SaSA.h"
#include "PRO.h"
#include "BDFO.h"
#include "SFO.h"

// Function Prototypes
void enforce_bound_constraints(Optimizer *opt);

Optimizer* general_init(int dim, int population_size, int max_iter, double* bounds, const char* method) {
    if (dim <= 0 || population_size <= 0 || max_iter <= 0 || !bounds || !method) {
        fprintf(stderr, "general_init: Invalid input parameters\n");
        return NULL;
    }

    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    if (!opt) {
        fprintf(stderr, "general_init: Memory allocation failed for Optimizer\n");
        return NULL;
    }

    opt->dim = dim;
    opt->population_size = population_size;
    opt->max_iter = max_iter;

    opt->bounds = (double*)malloc(2 * dim * sizeof(double));
    if (!opt->bounds) {
        free(opt);
        fprintf(stderr, "general_init: Memory allocation failed for bounds\n");
        return NULL;
    }
    memcpy(opt->bounds, bounds, 2 * dim * sizeof(double));

    opt->population = (Solution*)malloc(population_size * sizeof(Solution));
    if (!opt->population) {
        free(opt->bounds);
        free(opt);
        fprintf(stderr, "general_init: Memory allocation failed for population\n");
        return NULL;
    }

    // Allocate a contiguous block of memory for population positions
    size_t positions_size = (size_t)dim * population_size * sizeof(double);
    double* population_positions = (double*)malloc(positions_size);
    if (!population_positions) {
        free(opt->population);
        free(opt->bounds);
        free(opt);
        fprintf(stderr, "general_init: Memory allocation failed for population_positions (%zu bytes)\n", positions_size);
        return NULL;
    }

    for (int i = 0; i < population_size; i++) {
        opt->population[i].position = population_positions + (i * dim);
        opt->population[i].fitness = INFINITY;
    }

    opt->best_solution.position = (double*)malloc(dim * sizeof(double));
    if (!opt->best_solution.position) {
        free(population_positions);
        free(opt->population);
        free(opt->bounds);
        free(opt);
        fprintf(stderr, "general_init: Memory allocation failed for best_solution.position\n");
        return NULL;
    }
    opt->best_solution.fitness = INFINITY;

    // Initialize population within bounds
    srand((unsigned int)time(NULL)); // Seed random number generator
    double rand_norm = 1.0 / RAND_MAX;
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            double min_bound = bounds[2 * d];
            double max_bound = bounds[2 * d + 1];
            opt->population[i].position[d] = min_bound + ((double)rand() * rand_norm) * (max_bound - min_bound);
        }
    }

    // Log allocations for debugging
    fprintf(stderr, "general_init: Allocated optimizer at %p\n", opt);
    fprintf(stderr, "general_init: Allocated population at %p\n", opt->population);
    fprintf(stderr, "general_init: Allocated population_positions at %p (size=%zu)\n", population_positions, positions_size);
    fprintf(stderr, "general_init: Allocated best_solution.position at %p\n", opt->best_solution.position);
    fprintf(stderr, "general_init: Allocated bounds at %p\n", opt->bounds);

    // Select Optimization Algorithm
    if (strcmp(method, "DISO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))DISO_optimize;
    } else if (strcmp(method, "GPC") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))GPC_optimize;
    } else if (strcmp(method, "GWCA") == 0) {
        opt->optimize = (void (*)(void*, ObjectiveFunction))GWCA_optimize;
    } else if (strcmp(method, "SPO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SPO_optimize;
    } else if (strcmp(method, "ANS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))ANS_optimize;
    } else if (strcmp(method, "BSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))BSA_optimize;
    } else if (strcmp(method, "CS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CS_optimize;
    } else if (strcmp(method, "DE") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))DE_optimize;
    } else if (strcmp(method, "EA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))EA_optimize;
    } else if (strcmp(method, "GA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))GA_optimize;
    } else if (strcmp(method, "HS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))HS_optimize;
    } else if (strcmp(method, "MA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))MA_optimize;
    } else if (strcmp(method, "SFS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SFS_optimize;
    } else if (strcmp(method, "GLS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))GLS_optimize;
    } else if (strcmp(method, "ILS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))ILS_optimize;
    } else if (strcmp(method, "SA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SA_optimize;
    } else if (strcmp(method, "TS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))TS_optimize;
    } else if (strcmp(method, "VNS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))VNS_optimize;
    } else if (strcmp(method, "AFSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))AFSA_optimize;
    } else if (strcmp(method, "ES") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))ES_optimize;
    } else if (strcmp(method, "HBO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))HBO_optimize;
    } else if (strcmp(method, "KA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))KA_optimize;
    } else if (strcmp(method, "PO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))PO_optimize;
    } else if (strcmp(method, "SO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SO_optimize;
    } else if (strcmp(method, "ADS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))ADS_optimize;
    } else if (strcmp(method, "AAA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))AAA_optimize;
    } else if (strcmp(method, "BCO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))BCO_optimize;
    } else if (strcmp(method, "BBM") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))BBM_optimize;
    } else if (strcmp(method, "CFO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CFO_optimize;
    } else if (strcmp(method, "CBO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CBO_optimize;
    } else if (strcmp(method, "CRO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CRO_optimize;
    } else if (strcmp(method, "CA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CA_optimize;
    } else if (strcmp(method, "CulA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CulA_optimize;
    } else if (strcmp(method, "FA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))FA_optimize;
    } else if (strcmp(method, "FFO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))FFO_optimize;
    } else if (strcmp(method, "SDS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SDS_optimize;
    } else if (strcmp(method, "KCA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))KCA_optimize;
    } else if (strcmp(method, "LS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))LS_optimize;
    } else if (strcmp(method, "WCA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))WCA_optimize;
    } else if (strcmp(method, "ACO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))ACO_optimize;
    } else if (strcmp(method, "ALO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))ALO_optimize;
    } else if (strcmp(method, "ABC") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))ABC_optimize;
    } else if (strcmp(method, "BA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))BA_optimize;
    } else if (strcmp(method, "CSO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CSO_optimize;
    } else if (strcmp(method, "CroSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CroSA_optimize;
    } else if (strcmp(method, "CO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CO_optimize;
    } else if (strcmp(method, "CucS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CucS_optimize;
    } else if (strcmp(method, "EHO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))EHO_optimize;
    } else if (strcmp(method, "DEA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))DEA_optimize;
    } else if (strcmp(method, "EPO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))EPO_optimize;
    } else if (strcmp(method, "FHO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))FHO_optimize;
    } else if (strcmp(method, "FirefA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))FirefA_optimize;
    } else if (strcmp(method, "FlyFO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))FlyFO_optimize;
    } else if (strcmp(method, "HHO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))HHO_optimize;
    } else if (strcmp(method, "GWO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))GWO_optimize;
    } else if (strcmp(method, "JA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))JA_optimize;
    } else if (strcmp(method, "KHO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))KHO_optimize;
    } else if (strcmp(method, "MfA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))MfA_optimize;
    } else if (strcmp(method, "LOA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))LOA_optimize;
    } else if (strcmp(method, "MSO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))MSO_optimize;
    } else if (strcmp(method, "MFO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))MFO_optimize;
    } else if (strcmp(method, "OSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))OSA_optimize;
    } else if (strcmp(method, "PuO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))PuO_optimize;
    } else if (strcmp(method, "RDA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))RDA_optimize;
    } else if (strcmp(method, "SFL") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SFL_optimize;
    } else if (strcmp(method, "SMO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SMO_optimize;
    } else if (strcmp(method, "SWO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SWO_optimize;
    } else if (strcmp(method, "SSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SSA_optimize;
    } else if (strcmp(method, "CMOA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CMOA_optimize;
    } else if (strcmp(method, "WO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))WO_optimize;
    } else if (strcmp(method, "WOA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))WOA_optimize;
    } else if (strcmp(method, "FSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))FSA_optimize;
    } else if (strcmp(method, "BBO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))BBO_optimize;
    } else if (strcmp(method, "ICA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))ICA_optimize;
    } else if (strcmp(method, "SOA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SOA_optimize;
    } else if (strcmp(method, "SEO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SEO_optimize;
    } else if (strcmp(method, "SLCO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SLCO_optimize;
    } else if (strcmp(method, "TLBO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))TLBO_optimize;
    } else if (strcmp(method, "DRA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))DRA_optimize;
    } else if (strcmp(method, "SGO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SGO_optimize;
    } else if (strcmp(method, "CheRO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CheRO_optimize;
    } else if (strcmp(method, "CSS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CSS_optimize;
    } else if (strcmp(method, "BHA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))BHA_optimize;
    } else if (strcmp(method, "EFO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))EFO_optimize;
    } else if (strcmp(method, "EVO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))EVO_optimize;
    } else if (strcmp(method, "LSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))LSA_optimize;
    } else if (strcmp(method, "SCA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SCA_optimize;
    } else if (strcmp(method, "SAO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SAO_optimize;
    } else if (strcmp(method, "TEO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))TEO_optimize;
    } else if (strcmp(method, "PVS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))PVS_optimize;
    } else if (strcmp(method, "TFWO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))TFWO_optimize;
    } else if (strcmp(method, "ARFO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))ARFO_optimize;
    } else if (strcmp(method, "FPA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))FPA_optimize;
    } else if (strcmp(method, "POA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))POA_optimize;
    } else if (strcmp(method, "IWO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))IWO_optimize;
    } else if (strcmp(method, "WPA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))WPA_optimize;
    } else if (strcmp(method, "BSO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))BSO_optimize;
    } else if (strcmp(method, "AMO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))AMO_optimize;
    } else if (strcmp(method, "COA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))COA_optimize;
    } else if (strcmp(method, "GlowSO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))GlowSO_optimize;
    } else if (strcmp(method, "GalSO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))GalSO_optimize;
    } else if (strcmp(method, "DHLO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))DHLO_optimize;
    } else if (strcmp(method, "HPO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))HPO_optimize;
    } else if (strcmp(method, "IWD") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))IWD_optimize;
    } else if (strcmp(method, "JOA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))JOA_optimize;
    } else if (strcmp(method, "WGMO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))WGMO_optimize;
    } else if (strcmp(method, "LoSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))LoSA_optimize;
    } else if (strcmp(method, "RMO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))RMO_optimize;
    } else if (strcmp(method, "PSO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))PSO_optimize;
    } else if (strcmp(method, "SaSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SaSA_optimize;
    } else if (strcmp(method, "PRO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))PRO_optimize;
    } else if (strcmp(method, "BDFO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))BDFO_optimize;
    } else if (strcmp(method, "SFO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SFO_optimize;
    } else {
        fprintf(stderr, "general_init: Unknown optimization method: %s\n", method);
        free(opt->best_solution.position);
        free(population_positions);
        free(opt->population);
        free(opt->bounds);
        free(opt);
        return NULL;
    }

    return opt;
}

// Calls the selected optimization algorithm
void general_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    if (opt == NULL || opt->optimize == NULL) {
        fprintf(stderr, "general_optimize: Error: Optimizer is not initialized properly.\n");
        return;
    }
    
    // Call the assigned optimization method
    opt->optimize(opt, objective_function);
}

// Enforce Bound Constraints for Population and Best Solution
void enforce_bound_constraints(Optimizer *opt) {
    if (!opt) {
        fprintf(stderr, "enforce_bound_constraints: Invalid optimizer\n");
        return;
    }
    for (int i = 0; i < opt->population_size; i++) {
        if (!opt->population[i].position) {
            fprintf(stderr, "enforce_bound_constraints: Null population[%d].position\n", i);
            continue;
        }
        for (int j = 0; j < opt->dim; j++) {
            if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                opt->population[i].position[j] = opt->bounds[2 * j];
            }
            if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                opt->population[i].position[j] = opt->bounds[2 * j + 1];
            }
        }
    }
    if (opt->best_solution.position) {
        for (int j = 0; j < opt->dim; j++) {
            if (opt->best_solution.position[j] < opt->bounds[2 * j]) {
                opt->best_solution.position[j] = opt->bounds[2 * j];
            }
            if (opt->best_solution.position[j] > opt->bounds[2 * j + 1]) {
                opt->best_solution.position[j] = opt->bounds[2 * j + 1];
            }
        }
    }
}

// Free Allocated Memory
void general_free(Optimizer* opt) {
    if (!opt) {
        fprintf(stderr, "general_free: NULL optimizer\n");
        return;
    }
    
    fprintf(stderr, "general_free: Freeing best_solution.position at %p\n", opt->best_solution.position);
    if (opt->best_solution.position) {
        free(opt->best_solution.position);
        opt->best_solution.position = NULL;
    }
    
    fprintf(stderr, "general_free: Freeing population_positions at %p\n", opt->population ? opt->population[0].position : NULL);
    if (opt->population && opt->population[0].position) {
        // Verify pointer integrity
        for (int i = 1; i < opt->population_size; i++) {
            if (opt->population[i].position != opt->population[0].position + (i * opt->dim)) {
                fprintf(stderr, "general_free: Warning: population[%d].position (%p) does not match expected offset\n",
                        i, opt->population[i].position);
            }
        }
        free(opt->population[0].position);  // Free the contiguous memory block
        for (int i = 0; i < opt->population_size; i++) {
            opt->population[i].position = NULL; // Prevent dangling pointers
        }
    }
    
    fprintf(stderr, "general_free: Freeing population at %p\n", opt->population);
    if (opt->population) {
        free(opt->population);
        opt->population = NULL;
    }
    
    fprintf(stderr, "general_free: Freeing bounds at %p\n", opt->bounds);
    if (opt->bounds) {
        free(opt->bounds);
        opt->bounds = NULL;
    }
    
    fprintf(stderr, "general_free: Freeing optimizer at %p\n", opt);
    free(opt);
}

// Retrieve the Best Solution
void get_best_solution(Optimizer* opt, double* best_position, double* best_fitness) {
    if (!opt || !best_position || !best_fitness) {
        fprintf(stderr, "get_best_solution: Invalid arguments\n");
        return;
    }
    *best_fitness = opt->best_solution.fitness;
    memcpy(best_position, opt->best_solution.position, opt->dim * sizeof(double));
}
