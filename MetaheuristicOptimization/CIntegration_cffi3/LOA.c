#include "LOA.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (stub, as in FirefA)
char *preprocess_kernel_source(const char *source);

void LOA_init_cl(LOACLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
    cl_int err;

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }
    if (!kernel_source || strlen(kernel_source) == 0) {
        fprintf(stderr, "Error: Kernel source is empty or null\n");
        exit(1);
    }

    // Create command queue with profiling
    cl_command_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };
    cl_ctx->queue = clCreateCommandQueueWithProperties(opt->context, opt->device, props, &err);
    check_cl_error(err, "clCreateCommandQueueWithProperties");
    cl_ctx->owns_queue = CL_TRUE;

    // Log queue properties
    cl_command_queue_properties actual_props;
    err = clGetCommandQueueInfo(cl_ctx->queue, CL_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &actual_props, NULL);
    check_cl_error(err, "clGetCommandQueueInfo");
    printf("Queue profiling enabled: %s\n", (actual_props & CL_QUEUE_PROFILING_ENABLE) ? "Yes" : "No");

    // Preprocess kernel source
    char *processed_source = preprocess_kernel_source(kernel_source);

    // Reuse OpenCL context and device
    cl_ctx->context = opt->context;
    cl_ctx->device = opt->device;

    // Create and build program
    cl_ctx->program = clCreateProgramWithSource(cl_ctx->context, 1, (const char **)&processed_source, NULL, &err);
    check_cl_error(err, "clCreateProgramWithSource");

    const char *build_options = "-cl-std=CL1.2 -cl-mad-enable";
    err = clBuildProgram(cl_ctx->program, 1, &cl_ctx->device, build_options, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(cl_ctx->program, cl_ctx->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size + 1);
        if (log) {
            clGetProgramBuildInfo(cl_ctx->program, cl_ctx->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            log[log_size] = '\0';
            fprintf(stderr, "Error building program: %d\nDetailed Build Log:\n%s\n", err, log);
            free(log);
        } else {
            fprintf(stderr, "Error building program: %d\nFailed to allocate memory for build log\n", err);
        }
        clReleaseProgram(cl_ctx->program);
        free(processed_source);
        exit(1);
    }

    free(processed_source);

    // Create kernels
    cl_ctx->init_pop_kernel = clCreateKernel(cl_ctx->program, "initialize_population", &err);
    check_cl_error(err, "clCreateKernel init_pop");
    cl_ctx->hunting_kernel = clCreateKernel(cl_ctx->program, "hunting_phase", &err);
    check_cl_error(err, "clCreateKernel hunting");
    cl_ctx->move_safe_kernel = clCreateKernel(cl_ctx->program, "move_to_safe_place_phase", &err);
    check_cl_error(err, "clCreateKernel move_safe");
    cl_ctx->roaming_kernel = clCreateKernel(cl_ctx->program, "roaming_phase", &err);
    check_cl_error(err, "clCreateKernel roaming");
    cl_ctx->mating_kernel = clCreateKernel(cl_ctx->program, "mating_phase", &err);
    check_cl_error(err, "clCreateKernel mating");
    cl_ctx->nomad_move_kernel = clCreateKernel(cl_ctx->program, "nomad_movement_phase", &err);
    check_cl_error(err, "clCreateKernel nomad_move");
    cl_ctx->pop_control_kernel = clCreateKernel(cl_ctx->program, "population_control_phase", &err);
    check_cl_error(err, "clCreateKernel pop_control");
}

void LOA_cleanup_cl(LOACLContext *cl_ctx) {
    if (cl_ctx->pop_control_kernel) clReleaseKernel(cl_ctx->pop_control_kernel);
    if (cl_ctx->nomad_move_kernel) clReleaseKernel(cl_ctx->nomad_move_kernel);
    if (cl_ctx->mating_kernel) clReleaseKernel(cl_ctx->mating_kernel);
    if (cl_ctx->roaming_kernel) clReleaseKernel(cl_ctx->roaming_kernel);
    if (cl_ctx->move_safe_kernel) clReleaseKernel(cl_ctx->move_safe_kernel);
    if (cl_ctx->hunting_kernel) clReleaseKernel(cl_ctx->hunting_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void LOA_init_context(LOAContext *ctx, Optimizer *opt, LOACLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;

    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Validate population size
    if (pop_size < LOA_PRIDE_SIZE) {
        fprintf(stderr, "Error: Population size (%d) must be at least %d\n", pop_size, LOA_PRIDE_SIZE);
        exit(1);
    }

    // Calculate number of prides and nomads, ensuring all lions are assigned
    ctx->nomad_size = (int)(LOA_NOMAD_RATIO * pop_size);
    int pride_pop = pop_size - ctx->nomad_size;
    ctx->num_prides = pride_pop / LOA_PRIDE_SIZE;
    if (pride_pop % LOA_PRIDE_SIZE != 0) {
        ctx->num_prides++;
    }
    if (ctx->num_prides * LOA_PRIDE_SIZE + ctx->nomad_size < pop_size) {
        ctx->nomad_size = pop_size - ctx->num_prides * LOA_PRIDE_SIZE;
    }

    // Allocate OpenCL buffers
    ctx->population = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer population");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->best_position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_position");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->prides = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, ctx->num_prides * LOA_PRIDE_SIZE * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer prides");
    ctx->pride_sizes = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, ctx->num_prides * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer pride_sizes");
    ctx->nomads = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, ctx->nomad_size * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer nomads");
    ctx->genders = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_uchar), NULL, &err);
    check_cl_error(err, "clCreateBuffer genders");
    ctx->temp_buffer = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer temp_buffer");
    ctx->females = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, LOA_PRIDE_SIZE * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer females");
    ctx->hunters = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, LOA_PRIDE_SIZE * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer hunters");
    ctx->non_hunters = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, LOA_PRIDE_SIZE * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer non_hunters");
    ctx->males = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, LOA_PRIDE_SIZE * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer males");
    ctx->mating_females = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, LOA_PRIDE_SIZE * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer mating_females");
    ctx->nomad_females = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, ctx->nomad_size * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer nomad_females");

    // Initialize random seeds
    cl_uint *seeds = (cl_uint *)malloc(pop_size * sizeof(cl_uint));
    srand(time(NULL));
    for (int i = 0; i < pop_size; i++) seeds[i] = rand();
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->random_seeds, CL_TRUE, 0, pop_size * sizeof(cl_uint), seeds, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer seeds");
    free(seeds);

    // Initialize bounds
    float *bounds_float = (float *)malloc(dim * 2 * sizeof(float));
    for (int i = 0; i < dim * 2; i++) bounds_float[i] = (float)opt->bounds[i];
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->bounds, CL_TRUE, 0, dim * 2 * sizeof(cl_float), bounds_float, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer bounds");
    free(bounds_float);

    // Initialize prides and nomads (CPU-side)
    int num_nomads = ctx->nomad_size;
    int *indices = (int *)malloc(pop_size * sizeof(int));
    for (int i = 0; i < pop_size; i++) indices[i] = i;
    for (int i = pop_size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    int *prides = (int *)malloc(ctx->num_prides * LOA_PRIDE_SIZE * sizeof(int));
    int *pride_sizes = (int *)malloc(ctx->num_prides * sizeof(int));
    int *nomads = (int *)malloc(num_nomads * sizeof(int));
    int assigned_lions = 0;

    // Assign lions to prides
    for (int i = 0; i < ctx->num_prides; i++) {
        int current_pride_size = LOA_PRIDE_SIZE;
        if (i == ctx->num_prides - 1) {
            int remaining = pop_size - num_nomads - (ctx->num_prides - 1) * LOA_PRIDE_SIZE;
            current_pride_size = (remaining > 0) ? remaining : LOA_PRIDE_SIZE;
        }
        pride_sizes[i] = current_pride_size;
        for (int j = 0; j < current_pride_size && assigned_lions < pop_size; j++) {
            prides[i * LOA_PRIDE_SIZE + j] = indices[num_nomads + assigned_lions];
            assigned_lions++;
        }
        // Fill remaining slots with -1 if pride is underfilled
        for (int j = current_pride_size; j < LOA_PRIDE_SIZE; j++) {
            prides[i * LOA_PRIDE_SIZE + j] = -1;
        }
    }

    // Assign nomads
    for (int i = 0; i < num_nomads && i < pop_size; i++) {
        nomads[i] = indices[i];
    }

    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->prides, CL_TRUE, 0, ctx->num_prides * LOA_PRIDE_SIZE * sizeof(cl_int), prides, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer prides");
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->pride_sizes, CL_TRUE, 0, ctx->num_prides * sizeof(cl_int), pride_sizes, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer pride_sizes");
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->nomads, CL_TRUE, 0, num_nomads * sizeof(cl_int), nomads, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer nomads");

    // Initialize genders using prides array before freeing
    unsigned char *genders = (unsigned char *)calloc(pop_size, sizeof(unsigned char));
    for (int p = 0; p < ctx->num_prides; p++) {
        int num_females = (int)(LOA_FEMALE_RATIO * pride_sizes[p]);
        for (int i = 0; i < num_females; i++) {
            int idx;
            do {
                idx = prides[p * LOA_PRIDE_SIZE + (rand() % pride_sizes[p])];
            } while (idx == -1);
            genders[idx] = 1;
        }
    }
    int num_nomad_females = (int)(LOA_FEMALE_RATIO * num_nomads);
    for (int i = 0; i < num_nomad_females; i++) {
        int idx = nomads[rand() % num_nomads];
        genders[idx] = 1;
    }
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->genders, CL_TRUE, 0, pop_size * sizeof(cl_uchar), genders, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer genders");

    // Free CPU-side memory
    free(prides);
    free(pride_sizes);
    free(nomads);
    free(indices);
    free(genders);
}

void LOA_cleanup_context(LOAContext *ctx, LOACLContext *cl_ctx) {
    if (ctx->nomad_females) clReleaseMemObject(ctx->nomad_females);
    if (ctx->mating_females) clReleaseMemObject(ctx->mating_females);
    if (ctx->males) clReleaseMemObject(ctx->males);
    if (ctx->non_hunters) clReleaseMemObject(ctx->non_hunters);
    if (ctx->hunters) clReleaseMemObject(ctx->hunters);
    if (ctx->females) clReleaseMemObject(ctx->females);
    if (ctx->temp_buffer) clReleaseMemObject(ctx->temp_buffer);
    if (ctx->genders) clReleaseMemObject(ctx->genders);
    if (ctx->nomads) clReleaseMemObject(ctx->nomads);
    if (ctx->pride_sizes) clReleaseMemObject(ctx->pride_sizes);
    if (ctx->prides) clReleaseMemObject(ctx->prides);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

void LOA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Lion Optimization Algorithm...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("LOA.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open LOA.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: LOA.cl is empty\n");
        fclose(fp);
        exit(1);
    }
    rewind(fp);
    char *source_str = (char *)malloc(source_size + 1);
    if (!source_str) {
        fprintf(stderr, "Error: Memory allocation failed for kernel source\n");
        fclose(fp);
        exit(1);
    }
    size_t read_size = fread(source_str, 1, source_size, fp);
    fclose(fp);
    if (read_size != source_size) {
        fprintf(stderr, "Error: Failed to read LOA.cl completely (%zu of %zu bytes)\n", read_size, source_size);
        free(source_str);
        exit(1);
    }
    source_str[source_size] = '\0';

    int valid_content = 0;
    for (size_t i = 0; i < source_size; i++) {
        if (!isspace(source_str[i]) && isprint(source_str[i])) {
            valid_content = 1;
            break;
        }
    }
    if (!valid_content) {
        fprintf(stderr, "Error: LOA.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    LOACLContext cl_ctx = {0};
    LOA_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    LOAContext ctx = {0};
    LOA_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * pop_size : MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_pop_time;
        double hunting_time;
        double move_safe_time;
        double roaming_time;
        double mating_time;
        double nomad_move_time;
        double pop_control_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;
    cl_mem best_fitness_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_fitness");

    // Initialize population
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_pop_kernel, 0, sizeof(cl_mem), &ctx.population);
    clSetKernelArg(cl_ctx.init_pop_kernel, 1, sizeof(cl_mem), &ctx.fitness);
    clSetKernelArg(cl_ctx.init_pop_kernel, 2, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 3, sizeof(cl_mem), &ctx.best_position);
    clSetKernelArg(cl_ctx.init_pop_kernel, 4, sizeof(cl_mem), &best_fitness_buf);
    clSetKernelArg(cl_ctx.init_pop_kernel, 5, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 6, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_pop_kernel, 7, sizeof(cl_int), &pop_size);

    size_t global_work_size = pop_size;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.init_pop_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
    check_cl_error(err, "clEnqueueNDRangeKernel init_pop");
    clFinish(cl_ctx.queue);

    cl_ulong start, end;
    if (init_event) {
        err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Warning: clGetEventProfilingInfo init_pop start failed: %d\n", err);
        } else {
            err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Warning: clGetEventProfilingInfo init_pop end failed: %d\n", err);
            } else {
                prof.init_pop_time += (end - start) / 1e6; // Convert ns to ms
            }
        }
        clReleaseEvent(init_event);
    }

    // Evaluate initial population on CPU
    cl_event read_event, write_event;
    float *positions = (float *)malloc(pop_size * dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer population");
    float *fitness = (float *)malloc(pop_size * sizeof(float));
    for (int i = 0; i < pop_size; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
        fitness[i] = (float)objective_function(pos);
        free(pos);
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer fitness");
    free(positions);

    if (read_event) {
        err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.read_time += (end - start) / 1e6;
            }
        }
        clReleaseEvent(read_event);
    }

    if (write_event) {
        err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.write_time += (end - start) / 1e6;
            }
        }
        clReleaseEvent(write_event);
    }

    func_count += pop_size;
    float best_fitness = fitness[0];
    int best_idx = 0;
    for (int i = 1; i < pop_size; i++) {
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            best_idx = i;
        }
    }
    ctx.best_fitness = best_fitness;
    free(fitness);

    positions = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer best_pos");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer best_pos");
    free(positions);

    if (read_event) {
        err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.read_time += (end - start) / 1e6;
            }
        }
        clReleaseEvent(read_event);
    }

    if (write_event) {
        err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.write_time += (end - start) / 1e6;
            }
        }
        clReleaseEvent(write_event);
    }

    err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &best_fitness, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer best_fitness");

    if (write_event) {
        err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.write_time += (end - start) / 1e6;
            }
        }
        clReleaseEvent(write_event);
    }

    // Main optimization loop
    while (func_count < max_evals) {
        iteration++;
        printf("Iteration %d\n", iteration);

        // Declare profiling variables at loop scope
        double hunting_time = 0.0;
        double move_safe_time = 0.0;
        double roaming_time = 0.0;
        double mating_time = 0.0;
        double nomad_move_time = 0.0;
        double pop_control_time = 0.0;

        // Process each pride
        for (int p = 0; p < ctx.num_prides; p++) {
            // Hunting phase
            cl_event hunting_event;
            clSetKernelArg(cl_ctx.hunting_kernel, 0, sizeof(cl_mem), &ctx.population);
            clSetKernelArg(cl_ctx.hunting_kernel, 1, sizeof(cl_mem), &ctx.fitness);
            clSetKernelArg(cl_ctx.hunting_kernel, 2, sizeof(cl_mem), &ctx.bounds);
            clSetKernelArg(cl_ctx.hunting_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
            clSetKernelArg(cl_ctx.hunting_kernel, 4, sizeof(cl_mem), &ctx.prides);
            clSetKernelArg(cl_ctx.hunting_kernel, 5, sizeof(cl_mem), &ctx.pride_sizes);
            clSetKernelArg(cl_ctx.hunting_kernel, 6, sizeof(cl_mem), &ctx.genders);
            clSetKernelArg(cl_ctx.hunting_kernel, 7, sizeof(cl_mem), &ctx.temp_buffer);
            clSetKernelArg(cl_ctx.hunting_kernel, 8, sizeof(cl_mem), &ctx.females);
            clSetKernelArg(cl_ctx.hunting_kernel, 9, sizeof(cl_mem), &ctx.hunters);
            clSetKernelArg(cl_ctx.hunting_kernel, 10, sizeof(cl_int), &p);
            clSetKernelArg(cl_ctx.hunting_kernel, 11, sizeof(cl_int), &dim);
            clSetKernelArg(cl_ctx.hunting_kernel, 12, sizeof(cl_int), &pop_size);

            size_t pride_work_size = LOA_PRIDE_SIZE;
            err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.hunting_kernel, 1, NULL, &pride_work_size, NULL, 0, NULL, &hunting_event);
            check_cl_error(err, "clEnqueueNDRangeKernel hunting");
            clFinish(cl_ctx.queue);

            if (hunting_event) {
                err = clGetEventProfilingInfo(hunting_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(hunting_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        hunting_time = (end - start) / 1e6;
                        prof.hunting_time += hunting_time;
                    }
                }
                clReleaseEvent(hunting_event);
            }

            // Move to safe place phase
            cl_event move_safe_event;
            clSetKernelArg(cl_ctx.move_safe_kernel, 0, sizeof(cl_mem), &ctx.population);
            clSetKernelArg(cl_ctx.move_safe_kernel, 1, sizeof(cl_mem), &ctx.fitness);
            clSetKernelArg(cl_ctx.move_safe_kernel, 2, sizeof(cl_mem), &ctx.bounds);
            clSetKernelArg(cl_ctx.move_safe_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
            clSetKernelArg(cl_ctx.move_safe_kernel, 4, sizeof(cl_mem), &ctx.prides);
            clSetKernelArg(cl_ctx.move_safe_kernel, 5, sizeof(cl_mem), &ctx.pride_sizes);
            clSetKernelArg(cl_ctx.move_safe_kernel, 6, sizeof(cl_mem), &ctx.genders);
            clSetKernelArg(cl_ctx.move_safe_kernel, 7, sizeof(cl_mem), &ctx.females);
            clSetKernelArg(cl_ctx.move_safe_kernel, 8, sizeof(cl_mem), &ctx.hunters);
            clSetKernelArg(cl_ctx.move_safe_kernel, 9, sizeof(cl_mem), &ctx.non_hunters);
            clSetKernelArg(cl_ctx.move_safe_kernel, 10, sizeof(cl_int), &p);
            clSetKernelArg(cl_ctx.move_safe_kernel, 11, sizeof(cl_int), &dim);
            clSetKernelArg(cl_ctx.move_safe_kernel, 12, sizeof(cl_int), &pop_size);

            err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.move_safe_kernel, 1, NULL, &pride_work_size, NULL, 0, NULL, &move_safe_event);
            check_cl_error(err, "clEnqueueNDRangeKernel move_safe");
            clFinish(cl_ctx.queue);

            if (move_safe_event) {
                err = clGetEventProfilingInfo(move_safe_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(move_safe_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        move_safe_time = (end - start) / 1e6;
                        prof.move_safe_time += move_safe_time;
                    }
                }
                clReleaseEvent(move_safe_event);
            }

            // Roaming phase
            cl_event roaming_event;
            clSetKernelArg(cl_ctx.roaming_kernel, 0, sizeof(cl_mem), &ctx.population);
            clSetKernelArg(cl_ctx.roaming_kernel, 1, sizeof(cl_mem), &ctx.fitness);
            clSetKernelArg(cl_ctx.roaming_kernel, 2, sizeof(cl_mem), &ctx.bounds);
            clSetKernelArg(cl_ctx.roaming_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
            clSetKernelArg(cl_ctx.roaming_kernel, 4, sizeof(cl_mem), &ctx.prides);
            clSetKernelArg(cl_ctx.roaming_kernel, 5, sizeof(cl_mem), &ctx.pride_sizes);
            clSetKernelArg(cl_ctx.roaming_kernel, 6, sizeof(cl_mem), &ctx.genders);
            clSetKernelArg(cl_ctx.roaming_kernel, 7, sizeof(cl_mem), &ctx.males);
            clSetKernelArg(cl_ctx.roaming_kernel, 8, sizeof(cl_int), &p);
            clSetKernelArg(cl_ctx.roaming_kernel, 9, sizeof(cl_int), &dim);
            clSetKernelArg(cl_ctx.roaming_kernel, 10, sizeof(cl_int), &pop_size);

            err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.roaming_kernel, 1, NULL, &pride_work_size, NULL, 0, NULL, &roaming_event);
            check_cl_error(err, "clEnqueueNDRangeKernel roaming");
            clFinish(cl_ctx.queue);

            if (roaming_event) {
                err = clGetEventProfilingInfo(roaming_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(roaming_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        roaming_time = (end - start) / 1e6;
                        prof.roaming_time += roaming_time;
                    }
                }
                clReleaseEvent(roaming_event);
            }

            // Mating phase
            cl_event mating_event;
            clSetKernelArg(cl_ctx.mating_kernel, 0, sizeof(cl_mem), &ctx.population);
            clSetKernelArg(cl_ctx.mating_kernel, 1, sizeof(cl_mem), &ctx.fitness);
            clSetKernelArg(cl_ctx.mating_kernel, 2, sizeof(cl_mem), &ctx.bounds);
            clSetKernelArg(cl_ctx.mating_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
            clSetKernelArg(cl_ctx.mating_kernel, 4, sizeof(cl_mem), &ctx.prides);
            clSetKernelArg(cl_ctx.mating_kernel, 5, sizeof(cl_mem), &ctx.pride_sizes);
            clSetKernelArg(cl_ctx.mating_kernel, 6, sizeof(cl_mem), &ctx.genders);
            clSetKernelArg(cl_ctx.mating_kernel, 7, sizeof(cl_mem), &ctx.females);
            clSetKernelArg(cl_ctx.mating_kernel, 8, sizeof(cl_mem), &ctx.males);
            clSetKernelArg(cl_ctx.mating_kernel, 9, sizeof(cl_mem), &ctx.mating_females);
            clSetKernelArg(cl_ctx.mating_kernel, 10, sizeof(cl_mem), &ctx.temp_buffer);
            clSetKernelArg(cl_ctx.mating_kernel, 11, sizeof(cl_int), &p);
            clSetKernelArg(cl_ctx.mating_kernel, 12, sizeof(cl_int), &dim);
            clSetKernelArg(cl_ctx.mating_kernel, 13, sizeof(cl_int), &pop_size);

            err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.mating_kernel, 1, NULL, &pride_work_size, NULL, 0, NULL, &mating_event);
            check_cl_error(err, "clEnqueueNDRangeKernel mating");
            clFinish(cl_ctx.queue);

            if (mating_event) {
                err = clGetEventProfilingInfo(mating_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(mating_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        mating_time = (end - start) / 1e6;
                        prof.mating_time += mating_time;
                    }
                }
                clReleaseEvent(mating_event);
            }
        }

        // Nomad movement phase
        cl_event nomad_move_event;
        clSetKernelArg(cl_ctx.nomad_move_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.nomad_move_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.nomad_move_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.nomad_move_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.nomad_move_kernel, 4, sizeof(cl_mem), &ctx.nomads);
        clSetKernelArg(cl_ctx.nomad_move_kernel, 5, sizeof(cl_int), &ctx.nomad_size);
        clSetKernelArg(cl_ctx.nomad_move_kernel, 6, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.nomad_move_kernel, 7, sizeof(cl_int), &pop_size);

        size_t nomad_work_size = ctx.nomad_size;
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.nomad_move_kernel, 1, NULL, &nomad_work_size, NULL, 0, NULL, &nomad_move_event);
        check_cl_error(err, "clEnqueueNDRangeKernel nomad_move");
        clFinish(cl_ctx.queue);

        if (nomad_move_event) {
            err = clGetEventProfilingInfo(nomad_move_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(nomad_move_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    nomad_move_time = (end - start) / 1e6;
                    prof.nomad_move_time += nomad_move_time;
                }
            }
            clReleaseEvent(nomad_move_event);
        }

        // Population control phase
        cl_event pop_control_event;
        clSetKernelArg(cl_ctx.pop_control_kernel, 0, sizeof(cl_mem), &ctx.pride_sizes);
        clSetKernelArg(cl_ctx.pop_control_kernel, 1, sizeof(cl_int), &ctx.num_prides);

        size_t pride_count_work_size = ctx.num_prides;
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.pop_control_kernel, 1, NULL, &pride_count_work_size, NULL, 0, NULL, &pop_control_event);
        check_cl_error(err, "clEnqueueNDRangeKernel pop_control");
        clFinish(cl_ctx.queue);

        if (pop_control_event) {
            err = clGetEventProfilingInfo(pop_control_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(pop_control_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    pop_control_time = (end - start) / 1e6;
                    prof.pop_control_time += pop_control_time;
                }
            }
            clReleaseEvent(pop_control_event);
        }

        // Evaluate population on CPU
        positions = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population");
        fitness = (float *)malloc(pop_size * sizeof(float));
        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            fitness[i] = (float)objective_function(pos);
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer fitness");
        free(positions);
        func_count += pop_size;

        if (read_event) {
            err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.read_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(read_event);
        }

        if (write_event) {
            err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.write_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(write_event);
        }

        // Update best solution
        float new_best = fitness[0];
        best_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (fitness[i] < new_best) {
                new_best = fitness[i];
                best_idx = i;
            }
        }
        if (new_best < ctx.best_fitness) {
            ctx.best_fitness = new_best;
            positions = (float *)malloc(dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer best_pos");
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer best_pos");
            free(positions);

            if (read_event) {
                err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        prof.read_time += (end - start) / 1e6;
                    }
                }
                clReleaseEvent(read_event);
            }

            if (write_event) {
                err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        prof.write_time += (end - start) / 1e6;
                    }
                }
                clReleaseEvent(write_event);
            }

            err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &new_best, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer best_fitness");

            if (write_event) {
                err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        prof.write_time += (end - start) / 1e6;
                    }
                }
                clReleaseEvent(write_event);
            }
        }
        free(fitness);

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.best_fitness);
        printf("Profiling (ms): Hunting = %.3f, Move Safe = %.3f, Roaming = %.3f, Mating = %.3f, Nomad Move = %.3f, Pop Control = %.3f\n",
               hunting_time, move_safe_time, roaming_time, mating_time, nomad_move_time, pop_control_time);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_pop_time);
    printf("Hunting Phase: %.3f\n", prof.hunting_time);
    printf("Move to Safe Place: %.3f\n", prof.move_safe_time);
    printf("Roaming Phase: %.3f\n", prof.roaming_time);
    printf("Mating Phase: %.3f\n", prof.mating_time);
    printf("Nomad Movement: %.3f\n", prof.nomad_move_time);
    printf("Population Control: %.3f\n", prof.pop_control_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Hunting Phase: %.3f\n", prof.hunting_time / prof.count);
        printf("Move to Safe Place: %.3f\n", prof.move_safe_time / prof.count);
        printf("Roaming Phase: %.3f\n", prof.roaming_time / prof.count);
        printf("Mating Phase: %.3f\n", prof.mating_time / prof.count);
        printf("Nomad Movement: %.3f\n", prof.nomad_move_time / prof.count);
        printf("Population Control: %.3f\n", prof.pop_control_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Copy best solution to optimizer
    float *best_pos = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer final best_pos");
    opt->best_solution.fitness = ctx.best_fitness;
    for (int i = 0; i < dim; i++) opt->best_solution.position[i] = best_pos[i];
    free(best_pos);

    if (read_event) {
        err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.read_time += (end - start) / 1e6;
            }
        }
        clReleaseEvent(read_event);
    }

    // Cleanup
    clReleaseMemObject(best_fitness_buf);
    LOA_cleanup_context(&ctx, &cl_ctx);
    LOA_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
