#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h> // AVX for SIMD
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
#include "CBO.h"
#include "CFO.h"
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
#include "BA.h"
#include "ABC.h"
#include "CSO.h"
#include "CroSA.h"
#include "CO.h"
#include "CucS.h"
#include "DEA.h"
#include "EHO.h"
#include "EPO.h"
#include "FHO.h"
#include "FlyFO.h"
#include "FirefA.h"
#include "HHO.h"
#include "GWO.h"
#include "JA.h"
#include "KHO.h"
#include "MfA.h"
#include "MSO.h"
#include "LOA.h"
#include "MFO.h"
#include "OSA.h"
#include "PuO.h"
#include "RDA.h"
#include "SFL.h"
#include "SMO.h"
#include "SSA.h"
#include "SWO.h"
#include "CMOA.h"
#include "WO.h"
#include "WOA.h"
#include "FSA.h"
#include "BBO.h"
#include "ICA.h"
#include "SEO.h"
#include "SLCO.h"
#include "TLBO.h"
#include "SGO.h"
#include "DRA.h"
#include "CSS.h"
#include "BHA.h"
#include "CheRO.h"
#include "EFO.h"
#include "EVO.h"
#include "SCA.h"
#include "LSA.h"
#include "SAO.h"
#include "TEO.h"
#include "PVS.h"
#include "TFWO.h"
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
#include "LoSA.h"
#include "WGMO.h"
#include "RMO.h"
#include "PSO.h"
#include "PRO.h"
#include "SaSA.h"
#include "SOA.h"
#include "BDFO.h"
#include "SFO.h"
#include <stdint.h>

// Define a state for the Xorshift128+ generator
typedef struct {
    uint64_t state[2]; // Two 64-bit state variables
} Xorshift128Plus;

// Initialize the Xorshift128+ generator with a seed
void xorshift128plus_init(Xorshift128Plus *rng, uint64_t seed1, uint64_t seed2) {
    rng->state[0] = seed1;
    rng->state[1] = seed2;
}

// Generate a random 64-bit number using Xorshift128+
uint64_t xorshift128plus_next(Xorshift128Plus *rng) {
    uint64_t s1 = rng->state[0];
    const uint64_t s0 = rng->state[1];
    rng->state[0] = s0;
    s1 ^= s1 << 23; // a
    rng->state[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26); // b, c
    return rng->state[1] + s0;
}

// Generate a random float in [0, 1)
double xorshift128plus_next_double(Xorshift128Plus *rng) {
    return (xorshift128plus_next(rng) >> 11) * (1.0 / (1LL << 53));
}

// Function Prototypes
void enforce_bound_constraints(Optimizer *opt);

// Initialize Xorshift128+ RNG
Xorshift128Plus rng;

Optimizer* general_init(int dim, int population_size, int max_iter, double* bounds, const char* method) {
    // Initialize the RNG with a seed (you can set a fixed seed or use a time-based seed)
    xorshift128plus_init(&rng, 123456789ULL, 987654321ULL);  // Example seed values
    
    // Calculate required memory size
    size_t total_size = sizeof(Optimizer) + (2 * dim * sizeof(double)) + 
                        (population_size * sizeof(Solution)) + 
                        (population_size * dim * sizeof(double)) + (dim * sizeof(double));
    
    // Allocate a single large memory block (aligned to 32 bytes)
    void* mem_block = _mm_malloc(total_size, 32);  // Align memory to cache line size (32 bytes)
    if (!mem_block) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    Optimizer* opt = (Optimizer*)mem_block;
    opt->dim = dim;
    opt->population_size = population_size;
    opt->max_iter = max_iter;

    // Assign memory locations within the block
    char* mem_cursor = (char*)mem_block + sizeof(Optimizer);
    opt->bounds = (double*)mem_cursor;
    mem_cursor += 2 * dim * sizeof(double);

    opt->population = (Solution*)mem_cursor;
    mem_cursor += population_size * sizeof(Solution);

    double* population_positions = (double*)mem_cursor;
    mem_cursor += population_size * dim * sizeof(double);

    opt->best_solution.position = (double*)mem_cursor;

    // Copy bounds
    memcpy(opt->bounds, bounds, 2 * dim * sizeof(double));

    // Assign position pointers with cache locality in mind
    #pragma omp parallel for
    for (int i = 0; i < population_size; i++) {
        opt->population[i].position = population_positions + (i * dim);
    }

    opt->best_solution.fitness = INFINITY;

    // Initialize population within bounds with optimized random number generation
    #pragma omp parallel for
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            double min_bound = bounds[2 * d];
            double max_bound = bounds[2 * d + 1];
            // Use the fast Xorshift128+ RNG to generate random values
            opt->population[i].position[d] = min_bound + xorshift128plus_next_double(&rng) * (max_bound - min_bound);
        }
        opt->population[i].fitness = INFINITY;
    }

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
	} else if (strcmp(method, "CBO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CBO_optimize;
	} else if (strcmp(method, "CFO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CFO_optimize;
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
	} else if (strcmp(method, "BA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))BA_optimize;
	} else if (strcmp(method, "ABC") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))ABC_optimize;
	} else if (strcmp(method, "CSO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CSO_optimize;
	} else if (strcmp(method, "CroSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CroSA_optimize;
	} else if (strcmp(method, "CO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CO_optimize;
	} else if (strcmp(method, "CucS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CucS_optimize;
	} else if (strcmp(method, "DEA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))DEA_optimize;
	} else if (strcmp(method, "EHO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))EHO_optimize;
	} else if (strcmp(method, "EPO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))EPO_optimize;
	} else if (strcmp(method, "FHO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))FHO_optimize;
	} else if (strcmp(method, "FlyFO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))FlyFO_optimize;
	} else if (strcmp(method, "FirefA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))FirefA_optimize;
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
	} else if (strcmp(method, "MSO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))MSO_optimize;
	} else if (strcmp(method, "LOA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))LOA_optimize;
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
	} else if (strcmp(method, "SSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SSA_optimize;
	} else if (strcmp(method, "SWO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SWO_optimize;
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
	} else if (strcmp(method, "SEO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SEO_optimize;
	} else if (strcmp(method, "SLCO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SLCO_optimize;
    } else if (strcmp(method, "TLBO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))TLBO_optimize;
    } else if (strcmp(method, "SGO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SGO_optimize;
    } else if (strcmp(method, "DRA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))DRA_optimize;
    } else if (strcmp(method, "CSS") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CSS_optimize;
    } else if (strcmp(method, "BHA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))BHA_optimize;
    } else if (strcmp(method, "CheRO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CheRO_optimize;
    } else if (strcmp(method, "EFO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))EFO_optimize;
    } else if (strcmp(method, "EVO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))EVO_optimize;
    } else if (strcmp(method, "SCA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SCA_optimize;
    } else if (strcmp(method, "LSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))LSA_optimize;
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
    } else if (strcmp(method, "LoSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))LoSA_optimize;
    } else if (strcmp(method, "WGMO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))WGMO_optimize;
    } else if (strcmp(method, "RMO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))RMO_optimize;
    } else if (strcmp(method, "PSO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))PSO_optimize;
    } else if (strcmp(method, "PRO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))PRO_optimize;
    } else if (strcmp(method, "SaSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SaSA_optimize;
    } else if (strcmp(method, "SOA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SOA_optimize;
    } else if (strcmp(method, "BDFO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))BDFO_optimize;
    } else if (strcmp(method, "SFO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SFO_optimize;
    } else {
        fprintf(stderr, "Unknown optimization method: %s\n", method);
        _mm_free(mem_block);
        exit(EXIT_FAILURE);
    }

    return opt;
}

// Calls the selected optimization algorithm
void general_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    if (opt == NULL || opt->optimize == NULL) {
        fprintf(stderr, "Error: Optimizer is not initialized properly.\n");
        exit(EXIT_FAILURE);
    }
    
    // Call the assigned optimization method (DISO or GPC)
    opt->optimize(opt, objective_function);
}

// Enforce Bound Constraints using OpenMP
void enforce_bound_constraints(Optimizer *opt) {
    #pragma omp parallel for
    for (int i = 0; i < opt->population_size; i++) {
        for (int j = 0; j < opt->dim; j++) {
            if (opt->population[i].position[j] < opt->bounds[2 * j]) {
                opt->population[i].position[j] = opt->bounds[2 * j];
            }
            if (opt->population[i].position[j] > opt->bounds[2 * j + 1]) {
                opt->population[i].position[j] = opt->bounds[2 * j + 1];
            }
        }
    }

    #pragma omp parallel for
    for (int j = 0; j < opt->dim; j++) {
        if (opt->best_solution.position[j] < opt->bounds[2 * j]) {
            opt->best_solution.position[j] = opt->bounds[2 * j];
        }
        if (opt->best_solution.position[j] > opt->bounds[2 * j + 1]) {
            opt->best_solution.position[j] = opt->bounds[2 * j + 1];
        }
    }
}

void general_free(Optimizer* opt) {
    if (!opt) return;
    _mm_free(opt);
}

void get_best_solution(Optimizer* opt, double* best_position, double* best_fitness) {
    if (!opt || !best_position || !best_fitness) return;
    *best_fitness = opt->best_solution.fitness;
    memcpy(best_position, opt->best_solution.position, opt->dim * sizeof(double));
}
