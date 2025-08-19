#include "PRO.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (placeholder, can be extended if needed)
char *preprocess_kernel_source(const char *source);

void PRO_init_cl(PROCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->stimulate_kernel = clCreateKernel(cl_ctx->program, "stimulate_behaviors", &err);
    check_cl_error(err, "clCreateKernel stimulate_behaviors");
    cl_ctx->reschedule_kernel = clCreateKernel(cl_ctx->program, "reschedule", &err);
    check_cl_error(err, "clCreateKernel reschedule");
}

void PRO_cleanup_cl(PROCLContext *cl_ctx) {
    if (cl_ctx->reschedule_kernel) clReleaseKernel(cl_ctx->reschedule_kernel);
    if (cl_ctx->stimulate_kernel) clReleaseKernel(cl_ctx->stimulate_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void PRO_init_context(PROContext *ctx, Optimizer *opt, PROCLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;

    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->population = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer population");
    ctx->schedules = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer schedules");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->best_position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_position");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");

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
}

void PRO_cleanup_context(PROContext *ctx, PROCLContext *cl_ctx) {
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->schedules) clReleaseMemObject(ctx->schedules);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

// CPU-based behavior selection
void select_behaviors(Optimizer *opt, int i, int current_eval, float *schedules, int *selected_behaviors, int *landa) {
    float tau = (float)current_eval / PRO_MAX_EVALUATIONS;
    float selection_rate = expf(-(1.0f - tau));

    int *indices = (int *)malloc(opt->dim * sizeof(int));
    float *sched_float = (float *)malloc(opt->dim * sizeof(float));
    for (int j = 0; j < opt->dim; j++) {
        indices[j] = j;
        sched_float[j] = schedules[i * opt->dim + j];
    }

    // Use quicksort for descending order
    quicksort_with_indices_pro(sched_float, indices, 0, opt->dim - 1);

    *landa = (int)ceilf(opt->dim * ((float)rand() / RAND_MAX) * selection_rate);
    if (*landa > opt->dim) *landa = opt->dim;
    if (*landa < 1) *landa = 1;

    for (int j = 0; j < *landa; j++) {
        selected_behaviors[j] = indices[j];
    }

    free(indices);
    free(sched_float);
}

// CPU-based reinforcement
void apply_reinforcement(Optimizer *opt, int i, int *selected_behaviors, int landa, float *schedules, float *new_solution, float new_fitness, PROContext *ctx, PROCLContext *cl_ctx) {
    float current_fitness = opt->population[i].fitness;
    cl_int err;

    if (new_fitness < current_fitness) {
        // Positive Reinforcement
        for (int j = 0; j < landa; j++) {
            int idx = selected_behaviors[j];
            schedules[i * opt->dim + idx] += schedules[i * opt->dim + idx] * (REINFORCEMENT_RATE / 2.0f);
        }
        // Update population
        err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->population, CL_TRUE, i * opt->dim * sizeof(cl_float), opt->dim * sizeof(cl_float), new_solution, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer population");
        opt->population[i].fitness = new_fitness;

        // Update best solution
        if (new_fitness < ctx->best_fitness) {
            ctx->best_fitness = new_fitness;
            err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->best_position, CL_TRUE, 0, opt->dim * sizeof(cl_float), new_solution, 0, NULL, NULL);
            check_cl_error(err, "clEnqueueWriteBuffer best_position");
            opt->best_solution.fitness = new_fitness;
            for (int j = 0; j < opt->dim; j++) {
                opt->best_solution.position[j] = new_solution[j];
            }
        }
    } else {
        // Negative Reinforcement
        for (int j = 0; j < landa; j++) {
            int idx = selected_behaviors[j];
            schedules[i * opt->dim + idx] -= schedules[i * opt->dim + idx] * REINFORCEMENT_RATE;
        }
    }
}

// Quicksort implementation with indices
void quicksort_with_indices_pro(float *arr, int *indices, int low, int high) {
    if (low < high) {
        float pivot = arr[indices[high]];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (arr[indices[j]] >= pivot) {
                i++;
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
        int temp = indices[i + 1];
        indices[i + 1] = indices[high];
        indices[high] = temp;
        int pi = i + 1;

        quicksort_with_indices_pro(arr, indices, low, pi - 1);
        quicksort_with_indices_pro(arr, indices, pi + 1, high);
    }
}

void PRO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU PRO Algorithm optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("PRO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open PRO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: PRO.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read PRO.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: PRO.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    PROCLContext cl_ctx = {0};
    PRO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    PROContext ctx = {0};
    PRO_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;

    // Compute max evaluations
    int max_evals = PRO_MAX_EVALUATIONS;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_pop_time;
        double stimulate_time;
        double reschedule_time;
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
    clSetKernelArg(cl_ctx.init_pop_kernel, 1, sizeof(cl_mem), &ctx.schedules);
    clSetKernelArg(cl_ctx.init_pop_kernel, 2, sizeof(cl_mem), &ctx.fitness);
    clSetKernelArg(cl_ctx.init_pop_kernel, 3, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 4, sizeof(cl_mem), &ctx.best_position);
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

    // Additional buffers for stimulation phase
    cl_mem new_population = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer new_population");
    cl_mem selected_behaviors_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, dim * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer selected_behaviors");
    cl_mem other_position_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer other_position");

    // Main optimization loop
    float *schedules = (float *)malloc(pop_size * dim * sizeof(float));
    float *new_solution = (float *)malloc(dim * sizeof(float));
    int *selected_behaviors = (int *)malloc(dim * sizeof(int));
    int *pop_indices = (int *)malloc(pop_size * sizeof(int));
    float *fitness_values = (float *)malloc(pop_size * sizeof(float));
    float *temp_pos = (float *)malloc(dim * sizeof(float));
    float *temp_sched = (float *)malloc(dim * sizeof(float));

    while (func_count < max_evals) {
        iteration++;
        printf("Iteration %d\n", iteration);

        // Read schedules and fitness for CPU-based selection
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.schedules, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), schedules, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer schedules");
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness_values, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer fitness");

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

        // Sort population by fitness
        for (int i = 0; i < pop_size; i++) {
            pop_indices[i] = i;
            fitness_values[i] = opt->population[i].fitness = fitness[i];
        }
        quicksort_with_indices_pro(fitness_values, pop_indices, 0, pop_size - 1);

        // Apply sorted order
        positions = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer population");
        for (int i = 0; i < pop_size; i++) {
            if (pop_indices[i] != i) {
                memcpy(temp_pos, &positions[i * dim], dim * sizeof(float));
                memcpy(temp_sched, &schedules[i * dim], dim * sizeof(float));
                float temp_fitness = fitness_values[i];

                memcpy(&positions[i * dim], &positions[pop_indices[i] * dim], dim * sizeof(float));
                memcpy(&schedules[i * dim], &schedules[pop_indices[i] * dim], dim * sizeof(float));
                opt->population[i].fitness = fitness_values[pop_indices[i]];
                fitness_values[i] = fitness_values[pop_indices[i]];

                memcpy(&positions[pop_indices[i] * dim], temp_pos, dim * sizeof(float));
                memcpy(&schedules[pop_indices[i] * dim], temp_sched, dim * sizeof(float));
                fitness_values[pop_indices[i]] = temp_fitness;

                pop_indices[pop_indices[i]] = pop_indices[i];
                pop_indices[i] = i;
            }
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer population");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.schedules, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), schedules, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer schedules");

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

        for (int i = 0; i < pop_size; i++) {
            // Select another individual
            int k = pop_size - 1;
            if (i < pop_size - 1) {
                k = i + 1 + (int)(((double)rand() / RAND_MAX) * (pop_size - i - 1));
            }

            // Behavior selection on CPU
            int landa;
            select_behaviors(opt, i, func_count, schedules, selected_behaviors, &landa);

            // Write selected behaviors and other position
            err = clEnqueueWriteBuffer(cl_ctx.queue, selected_behaviors_buf, CL_TRUE, 0, landa * sizeof(cl_int), selected_behaviors, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer selected_behaviors");
            err = clEnqueueWriteBuffer(cl_ctx.queue, other_position_buf, CL_TRUE, 0, dim * sizeof(cl_float), &positions[k * dim], 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer other_position");

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

            // Stimulate behaviors on GPU
            cl_event stimulate_event;
            float tau = (float)func_count / PRO_MAX_EVALUATIONS;
            clSetKernelArg(cl_ctx.stimulate_kernel, 0, sizeof(cl_mem), &ctx.population);
            clSetKernelArg(cl_ctx.stimulate_kernel, 1, sizeof(cl_mem), &ctx.schedules);
            clSetKernelArg(cl_ctx.stimulate_kernel, 2, sizeof(cl_mem), &ctx.best_position);
            clSetKernelArg(cl_ctx.stimulate_kernel, 3, sizeof(cl_mem), &other_position_buf);
            clSetKernelArg(cl_ctx.stimulate_kernel, 4, sizeof(cl_mem), &new_population);
            clSetKernelArg(cl_ctx.stimulate_kernel, 5, sizeof(cl_mem), &ctx.random_seeds);
            clSetKernelArg(cl_ctx.stimulate_kernel, 6, sizeof(cl_mem), &selected_behaviors_buf);
            clSetKernelArg(cl_ctx.stimulate_kernel, 7, sizeof(cl_int), &landa);
            clSetKernelArg(cl_ctx.stimulate_kernel, 8, sizeof(cl_float), &tau);
            clSetKernelArg(cl_ctx.stimulate_kernel, 9, sizeof(cl_int), &dim);
            clSetKernelArg(cl_ctx.stimulate_kernel, 10, sizeof(cl_int), &pop_size);
            clSetKernelArg(cl_ctx.stimulate_kernel, 11, sizeof(cl_mem), &ctx.bounds);

            err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.stimulate_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &stimulate_event);
            check_cl_error(err, "clEnqueueNDRangeKernel stimulate_behaviors");
            clFinish(cl_ctx.queue);

            if (stimulate_event) {
                err = clGetEventProfilingInfo(stimulate_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                if (err == CL_SUCCESS) {
                    err = clGetEventProfilingInfo(stimulate_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                    if (err == CL_SUCCESS) {
                        prof.stimulate_time += (end - start) / 1e6;
                    }
                }
                clReleaseEvent(stimulate_event);
            }

            // Evaluate new solution on CPU
            err = clEnqueueReadBuffer(cl_ctx.queue, new_population, CL_TRUE, i * dim * sizeof(cl_float), dim * sizeof(cl_float), new_solution, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer new_population");
            double *new_pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) new_pos[j] = new_solution[j];
            float new_fitness = (float)objective_function(new_pos);
            free(new_pos);
            func_count++;

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

            // Apply reinforcement on CPU
            apply_reinforcement(opt, i, selected_behaviors, landa, schedules, new_solution, new_fitness, &ctx, &cl_ctx);

            // Write updated schedules
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.schedules, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), schedules, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer schedules");

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

            if (func_count >= max_evals) break;
        }

        // Update fitness values after rescheduling
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness_values, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer fitness");
        for (int i = 0; i < pop_size; i++) {
            if (fitness_values[i] == INFINITY) {
                double *pos = (double *)malloc(dim * sizeof(double));
                for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
                fitness_values[i] = (float)objective_function(pos);
                free(pos);
                func_count++;
            }
            opt->population[i].fitness = fitness_values[i];
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness_values, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer fitness");

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

        free(positions);
        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.best_fitness);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Population: %.3f\n", prof.init_pop_time);
    printf("Stimulate Behaviors: %.3f\n", prof.stimulate_time);
    printf("Reschedule: %.3f\n", prof.reschedule_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Stimulate Behaviors: %.3f\n", prof.stimulate_time / prof.count);
        printf("Reschedule: %.3f\n", prof.reschedule_time / prof.count);
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
    clReleaseMemObject(new_population);
    clReleaseMemObject(selected_behaviors_buf);
    clReleaseMemObject(other_position_buf);
    clReleaseMemObject(best_fitness_buf);
    free(schedules);
    free(new_solution);
    free(selected_behaviors);
    free(pop_indices);
    free(fitness_values);
    free(temp_pos);
    free(temp_sched);
    PRO_cleanup_context(&ctx, &cl_ctx);
    PRO_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
