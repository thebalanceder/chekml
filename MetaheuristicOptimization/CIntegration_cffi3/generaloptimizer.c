/* generaloptimizer.c */
#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
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
#include "CroSA.h"
#include "CSO.h"
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
//#include "SAO.h"
#include "TEO.h"
//#include "PVS.h"
#include "TFWO.h"
#include "ARFO.h"
#include "FPA.h"
#include "POA.h"
//#include "IWO.h"
#include "WPA.h"
//#include "BSO.h"
#include "AMO.h"
//#include "COA.h"
#include "GlowSO.h"
//#include "GalSO.h"
#include "DHLO.h"
#include "HPO.h"
#include "IWD.h"
//#include "JOA.h"
#include "WGMO.h"
//#include "LoSA.h"
//#include "RMO.h"
//#include "PSO.h"
#include "PRO.h"
#include "SaSA.h"
#include "SOA.h"
#include <CL/cl.h>

// Declare OpenCL variables
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_mem population_buffer, fitness_buffer;

double rand_double(double min, double max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

Optimizer* general_init(int dim, int population_size, int max_iter, double* bounds, const char* method) {
    Optimizer* opt = (Optimizer*)malloc(sizeof(Optimizer));
    if (!opt) {
        fprintf(stderr, "Memory allocation failed for Optimizer\n");
        exit(EXIT_FAILURE);
    }

    opt->dim = dim;
    opt->population_size = population_size;
    opt->max_iter = max_iter;

    opt->bounds = (double*)malloc(2 * dim * sizeof(double));
    if (!opt->bounds) {
        free(opt);
        fprintf(stderr, "Memory allocation failed for bounds\n");
        exit(EXIT_FAILURE);
    }
    memcpy(opt->bounds, bounds, 2 * dim * sizeof(double));

    opt->population = (Solution*)malloc(population_size * sizeof(Solution));
    if (!opt->population) {
        free(opt->bounds);
        free(opt);
        fprintf(stderr, "Memory allocation failed for population\n");
        exit(EXIT_FAILURE);
    }

    double* population_positions = (double*)malloc(dim * population_size * sizeof(double));
    if (!population_positions) {
        free(opt->population);
        free(opt->bounds);
        free(opt);
        fprintf(stderr, "Memory allocation failed for population positions\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < population_size; i++) {
        opt->population[i].position = population_positions + (i * dim);
    }

    opt->best_solution.position = (double*)malloc(dim * sizeof(double));
    if (!opt->best_solution.position) {
        free(population_positions);
        free(opt->population);
        free(opt->bounds);
        free(opt);
        fprintf(stderr, "Memory allocation failed for best_solution.position\n");
        exit(EXIT_FAILURE);
    }
    opt->best_solution.fitness = INFINITY;

    double rand_norm = 1.0 / RAND_MAX;
    for (int i = 0; i < population_size; i++) {
        for (int d = 0; d < dim; d++) {
            double min_bound = bounds[2 * d];
            double max_bound = bounds[2 * d + 1];
            opt->population[i].position[d] = min_bound + ((double)rand() * rand_norm) * (max_bound - min_bound);
        }
        opt->population[i].fitness = INFINITY;
    }

    opt->context = NULL;
    opt->queue = NULL;
    opt->device = NULL;
    opt->population_buffer = NULL;
    opt->fitness_buffer = NULL;

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
	} else if (strcmp(method, "CroSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CroSA_optimize;
	} else if (strcmp(method, "CSO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))CSO_optimize;
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
    //} else if (strcmp(method, "SAO") == 0) {
    //    opt->optimize = (void (*)(void *, ObjectiveFunction))SAO_optimize;
    } else if (strcmp(method, "TEO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))TEO_optimize;
    //} else if (strcmp(method, "PVS") == 0) {
    //    opt->optimize = (void (*)(void *, ObjectiveFunction))PVS_optimize;
	} else if (strcmp(method, "TFWO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))TFWO_optimize;
    } else if (strcmp(method, "ARFO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))ARFO_optimize;
    } else if (strcmp(method, "FPA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))FPA_optimize;
    } else if (strcmp(method, "POA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))POA_optimize;
    //} else if (strcmp(method, "IWO") == 0) {
    //    opt->optimize = (void (*)(void *, ObjectiveFunction))IWO_optimize;
	} else if (strcmp(method, "WPA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))WPA_optimize;
    //} else if (strcmp(method, "BSO") == 0) {
    //    opt->optimize = (void (*)(void *, ObjectiveFunction))BSO_optimize;
    } else if (strcmp(method, "AMO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))AMO_optimize;
    //} else if (strcmp(method, "COA") == 0) {
    //    opt->optimize = (void (*)(void *, ObjectiveFunction))COA_optimize;
    } else if (strcmp(method, "GlowSO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))GlowSO_optimize;
    //} else if (strcmp(method, "GalSO") == 0) {
    //    opt->optimize = (void (*)(void *, ObjectiveFunction))GalSO_optimize;
    } else if (strcmp(method, "DHLO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))DHLO_optimize;
    } else if (strcmp(method, "HPO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))HPO_optimize;
    } else if (strcmp(method, "IWD") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))IWD_optimize;
    //} else if (strcmp(method, "JOA") == 0) {
    //    opt->optimize = (void (*)(void *, ObjectiveFunction))JOA_optimize;
    } else if (strcmp(method, "WGMO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))WGMO_optimize;
    //} else if (strcmp(method, "LoSA") == 0) {
    //    opt->optimize = (void (*)(void *, ObjectiveFunction))LoSA_optimize;
    //} else if (strcmp(method, "RMO") == 0) {
    //    opt->optimize = (void (*)(void *, ObjectiveFunction))RMO_optimize;
    //} else if (strcmp(method, "PSO") == 0) {
    //    opt->optimize = (void (*)(void *, ObjectiveFunction))PSO_optimize;
    } else if (strcmp(method, "PRO") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))PRO_optimize;
    } else if (strcmp(method, "SaSA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SaSA_optimize;
    } else if (strcmp(method, "SOA") == 0) {
        opt->optimize = (void (*)(void *, ObjectiveFunction))SOA_optimize;
    } else {
        fprintf(stderr, "Unknown optimization method: %s\n", method);
        general_free(opt);
        exit(EXIT_FAILURE);
    }

    return opt;
}

void general_optimize(Optimizer* opt, ObjectiveFunction objective_function) {
    if (opt == NULL || opt->optimize == NULL) {
        fprintf(stderr, "Error: Optimizer is not initialized properly\n");
        exit(EXIT_FAILURE);
    }

    initialize_opencl(opt);
    create_buffers(opt);
    opt->optimize(opt, objective_function);
    cleanup_opencl(opt);
}

void enforce_bound_constraints(Optimizer *opt) {
    if (!opt || !opt->population || !opt->bounds) return;
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
    free(opt->best_solution.position);
    if (opt->population && opt->population[0].position) {
        free(opt->population[0].position);
    }
    free(opt->population);
    free(opt->bounds);
    free(opt);
}

void get_best_solution(Optimizer* opt, double* best_position, double* best_fitness) {
    if (!opt || !best_position || !best_fitness) return;
    *best_fitness = opt->best_solution.fitness;
    memcpy(best_position, opt->best_solution.position, opt->dim * sizeof(double));
}

void initialize_opencl(Optimizer *opt) {
    cl_int err;

    cl_platform_id platforms[10];
    cl_uint num_platforms;
    err = clGetPlatformIDs(10, platforms, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "Error getting platforms: %d\n", err);
        exit(EXIT_FAILURE);
    }

    platform = NULL;
    for (cl_uint i = 0; i < num_platforms; i++) {
        char platform_name[1024];
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error getting platform name: %d\n", err);
            exit(EXIT_FAILURE);
        }
        printf("Available platform: %s\n", platform_name);
        if (strstr(platform_name, "Intel(R) OpenCL HD Graphics") != NULL) {
            platform = platforms[i];
            break;
        }
    }

    if (!platform) {
        fprintf(stderr, "Error: Could not find 'Intel(R) OpenCL HD Graphics' platform\n");
        exit(EXIT_FAILURE);
    }

    cl_device_id devices[10];
    cl_uint num_devices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 10, devices, &num_devices);
    if (err != CL_SUCCESS || num_devices == 0) {
        fprintf(stderr, "Error getting devices: %d\n", err);
        exit(EXIT_FAILURE);
    }
    device = devices[0];

    char device_name[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    cl_device_type device_type;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    printf("Selected device: %s\n", device_name);
    if (device_type & CL_DEVICE_TYPE_GPU) {
        printf("The selected device is a GPU.\n");
    } else {
        printf("The selected device is not a GPU.\n");
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS || !context) {
        fprintf(stderr, "Error creating context: %d\n", err);
        exit(EXIT_FAILURE);
    }

    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    if (err != CL_SUCCESS || !queue) {
        fprintf(stderr, "Error creating command queue: %d\n", err);
        clReleaseContext(context);
        exit(EXIT_FAILURE);
    }

    opt->context = context;
    opt->queue = queue;
    opt->device = device;
}

void create_buffers(Optimizer* opt) {
    cl_int err;

    float* population_float = (float*)malloc(opt->dim * opt->population_size * sizeof(float));
    if (!population_float) {
        fprintf(stderr, "Error: Memory allocation failed for population_float\n");
        cleanup_opencl(opt);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < opt->dim * opt->population_size; i++) {
        population_float[i] = (float)opt->population[0].position[i];
    }

    population_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
                                       opt->dim * opt->population_size * sizeof(float), 
                                       NULL, &err);
    if (err != CL_SUCCESS || !population_buffer) {
        fprintf(stderr, "Error creating buffer for population: %d\n", err);
        free(population_float);
        cleanup_opencl(opt);
        exit(EXIT_FAILURE);
    }
    opt->population_buffer = population_buffer;

    err = clEnqueueWriteBuffer(opt->queue, opt->population_buffer, CL_TRUE, 0, 
                               opt->dim * opt->population_size * sizeof(float), 
                               population_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing population to buffer: %d\n", err);
        free(population_float);
        clReleaseMemObject(population_buffer);
        cleanup_opencl(opt);
        exit(EXIT_FAILURE);
    }
    free(population_float);

    fitness_buffer = clCreateBuffer(opt->context, CL_MEM_READ_WRITE, 
                                   opt->population_size * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !fitness_buffer) {
        fprintf(stderr, "Error creating buffer for fitness: %d\n", err);
        clReleaseMemObject(population_buffer);
        cleanup_opencl(opt);
        exit(EXIT_FAILURE);
    }
    opt->fitness_buffer = fitness_buffer;
}

void generate_points_on_gpu(Optimizer* opt, int points_per_dim, int total_points) {
    cl_int err;
    cl_program program = NULL;
    cl_kernel generate_kernel = NULL;
    cl_mem bounds_buffer = NULL;

    const char* kernel_source =
        "__kernel void generate_cc_points(\n"
        "    __global float* points,\n"
        "    __global const float* bounds,\n"
        "    const int dim,\n"
        "    const int points_per_dim,\n"
        "    const int total_points)\n"
        "{\n"
        "    int id = get_global_id(0);\n"
        "    if (id < total_points) {\n"
        "        int dim_idx = id / points_per_dim;\n"
        "        int pt_idx = id % points_per_dim;\n"
        "        float theta = 3.1415926535f * pt_idx / (points_per_dim - 1);\n"
        "        float cc_val = 0.5f * (1.0f - cos(theta));\n"
        "        float scale = bounds[2 * dim_idx + 1] - bounds[2 * dim_idx];\n"
        "        float offset = bounds[2 * dim_idx];\n"
        "        points[id] = offset + cc_val * scale;\n"
        "    }\n"
        "}\n";

    program = clCreateProgramWithSource(opt->context, 1, &kernel_source, NULL, &err);
    if (err != CL_SUCCESS || !program) {
        fprintf(stderr, "Error creating program: %d\n", err);
        exit(EXIT_FAILURE);
    }

    err = clBuildProgram(program, 1, &opt->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        if (log) {
            clGetProgramBuildInfo(program, opt->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            fprintf(stderr, "Error building program: %d\nBuild log:\n%s\n", err, log);
            free(log);
        }
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    generate_kernel = clCreateKernel(program, "generate_cc_points", &err);
    if (err != CL_SUCCESS || !generate_kernel) {
        fprintf(stderr, "Error creating generate kernel: %d\n", err);
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    float* bounds_float = (float*)malloc(2 * opt->dim * sizeof(float));
    if (!bounds_float) {
        fprintf(stderr, "Error: Memory allocation failed for bounds_float\n");
        clReleaseKernel(generate_kernel);
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < 2 * opt->dim; i++) {
        bounds_float[i] = (float)opt->bounds[i];
    }

    bounds_buffer = clCreateBuffer(opt->context, CL_MEM_READ_ONLY, 
                                   2 * opt->dim * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS || !bounds_buffer) {
        fprintf(stderr, "Error creating bounds buffer: %d\n", err);
        free(bounds_float);
        clReleaseKernel(generate_kernel);
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    err = clEnqueueWriteBuffer(opt->queue, bounds_buffer, CL_TRUE, 0, 
                               2 * opt->dim * sizeof(float), bounds_float, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error writing bounds buffer: %d\n", err);
        free(bounds_float);
        clReleaseMemObject(bounds_buffer);
        clReleaseKernel(generate_kernel);
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    err = clSetKernelArg(generate_kernel, 0, sizeof(cl_mem), &opt->population_buffer);
    err |= clSetKernelArg(generate_kernel, 1, sizeof(cl_mem), &bounds_buffer);
    err |= clSetKernelArg(generate_kernel, 2, sizeof(int), &opt->dim);
    err |= clSetKernelArg(generate_kernel, 3, sizeof(int), &points_per_dim);
    err |= clSetKernelArg(generate_kernel, 4, sizeof(int), &total_points);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting kernel args: %d\n", err);
        free(bounds_float);
        clReleaseMemObject(bounds_buffer);
        clReleaseKernel(generate_kernel);
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    size_t global_work_size = total_points;
    err = clEnqueueNDRangeKernel(opt->queue, generate_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error enqueuing generate kernel: %d\n", err);
        free(bounds_float);
        clReleaseMemObject(bounds_buffer);
        clReleaseKernel(generate_kernel);
        clReleaseProgram(program);
        exit(EXIT_FAILURE);
    }

    free(bounds_float);
    clReleaseMemObject(bounds_buffer);
    clReleaseKernel(generate_kernel);
    clReleaseProgram(program);
    clFinish(opt->queue);
}

void cleanup_opencl(Optimizer *opt) {
    if (opt->fitness_buffer) clReleaseMemObject(opt->fitness_buffer);
    if (opt->population_buffer) clReleaseMemObject(opt->population_buffer);
    if (opt->queue) clReleaseCommandQueue(opt->queue);
    if (opt->context) clReleaseContext(opt->context);
}
