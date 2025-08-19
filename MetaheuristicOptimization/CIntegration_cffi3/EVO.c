#include "EVO.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (placeholder, as in FirefA)
char *preprocess_kernel_source(const char *source);

void EVO_init_cl(EVOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->init_kernel = clCreateKernel(cl_ctx->program, "initialize_particles", &err);
    check_cl_error(err, "clCreateKernel init");
    cl_ctx->gradient_kernel = clCreateKernel(cl_ctx->program, "compute_gradient", &err);
    check_cl_error(err, "clCreateKernel gradient");
    cl_ctx->update_kernel = clCreateKernel(cl_ctx->program, "update_velocity_and_position", &err);
    check_cl_error(err, "clCreateKernel update");
}

void EVO_cleanup_cl(EVOCLContext *cl_ctx) {
    if (cl_ctx->update_kernel) clReleaseKernel(cl_ctx->update_kernel);
    if (cl_ctx->gradient_kernel) clReleaseKernel(cl_ctx->gradient_kernel);
    if (cl_ctx->init_kernel) clReleaseKernel(cl_ctx->init_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void EVO_init_context(EVOContext *ctx, Optimizer *opt, EVOCLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;

    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer position");
    ctx->velocity = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer velocity");
    ctx->gradient = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer gradient");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->best_position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_position");

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

void EVO_cleanup_context(EVOContext *ctx, EVOCLContext *cl_ctx) {
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->gradient) clReleaseMemObject(ctx->gradient);
    if (ctx->velocity) clReleaseMemObject(ctx->velocity);
    if (ctx->position) clReleaseMemObject(ctx->position);
}

// Quicksort implementation for sorting indices by fitness
static void quicksort_indices(float *fitness, int *indices, int low, int high) {
    if (low < high) {
        float pivot = fitness[indices[high]];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (fitness[indices[j]] <= pivot) {
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
        quicksort_indices(fitness, indices, low, pi - 1);
        quicksort_indices(fitness, indices, pi + 1, high);
    }
}

void EVO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Evolutionary Optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device ||
        !opt->population_buffer || !opt->fitness_buffer ||
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("EVO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open EVO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: EVO.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read EVO.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: EVO.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    EVOCLContext cl_ctx = {0};
    EVO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    EVOContext ctx = {0};
    EVO_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * pop_size : MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_time;
        double gradient_time;
        double update_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;

    // Initialize particles
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_kernel, 0, sizeof(cl_mem), &ctx.position);
    clSetKernelArg(cl_ctx.init_kernel, 1, sizeof(cl_mem), &ctx.velocity);
    clSetKernelArg(cl_ctx.init_kernel, 2, sizeof(cl_mem), &ctx.fitness);
    clSetKernelArg(cl_ctx.init_kernel, 3, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_kernel, 5, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_kernel, 6, sizeof(cl_int), &pop_size);

    size_t global_work_size = pop_size;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.init_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
    check_cl_error(err, "clEnqueueNDRangeKernel init");
    clFinish(cl_ctx.queue);

    cl_ulong start, end;
    if (init_event) {
        err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.init_time += (end - start) / 1e6; // Convert ns to ms
            }
        }
        clReleaseEvent(init_event);
    }

    // Evaluate initial population on CPU
    cl_event read_event, write_event;
    float *positions = (float *)malloc(pop_size * dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.position, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer position");
    float *fitness = (float *)malloc(pop_size * sizeof(float));
    for (int i = 0; i < pop_size; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
        fitness[i] = (float)objective_function(pos);
        free(pos);
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
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

    float *best_pos = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.position, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer best_pos");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer best_pos");

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

    // Main optimization loop
    while (func_count < max_evals) {
        iteration++;
        printf("Iteration %d\n", iteration);

        // Compute gradients (placeholder on CPU, as objective function is not GPU-compatible)
        float *gradients = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.position, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer position for gradient");
        for (int i = 0; i < pop_size; i++) {
            double x_plus[dim], x_minus[dim];
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            for (int j = 0; j < dim; j++) {
                x_plus[j] = pos[j];
                x_minus[j] = pos[j];
            }
            for (int j = 0; j < dim; j++) {
                x_plus[j] += EVO_LEARNING_RATE;
                x_minus[j] -= EVO_LEARNING_RATE;
                gradients[i * dim + j] = (float)((objective_function(x_plus) - objective_function(x_minus)) / (2 * EVO_LEARNING_RATE));
                x_plus[j] = pos[j];
                x_minus[j] = pos[j];
            }
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.gradient, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), gradients, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer gradient");
        free(gradients);

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

        // Update velocities and positions
        cl_event update_event;
        cl_float step_size = EVO_STEP_SIZE;
        cl_float momentum = EVO_MOMENTUM;
        clSetKernelArg(cl_ctx.update_kernel, 0, sizeof(cl_mem), &ctx.position);
        clSetKernelArg(cl_ctx.update_kernel, 1, sizeof(cl_mem), &ctx.velocity);
        clSetKernelArg(cl_ctx.update_kernel, 2, sizeof(cl_mem), &ctx.gradient);
        clSetKernelArg(cl_ctx.update_kernel, 3, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.update_kernel, 4, sizeof(cl_float), &step_size);
        clSetKernelArg(cl_ctx.update_kernel, 5, sizeof(cl_float), &momentum);
        clSetKernelArg(cl_ctx.update_kernel, 6, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.update_kernel, 7, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &update_event);
        check_cl_error(err, "clEnqueueNDRangeKernel update");
        clFinish(cl_ctx.queue);

        if (update_event) {
            err = clGetEventProfilingInfo(update_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(update_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.update_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(update_event);
        }

        // Evaluate fitness on CPU
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.position, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer position");
        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            fitness[i] = (float)objective_function(pos);
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, &write_event);
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

        func_count += pop_size;

        // Update best solution
        float new_best_fitness = fitness[0];
        int new_best_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (fitness[i] < new_best_fitness) {
                new_best_fitness = fitness[i];
                new_best_idx = i;
            }
        }
        if (new_best_fitness < ctx.best_fitness) {
            ctx.best_fitness = new_best_fitness;
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.position, CL_TRUE, new_best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
            check_cl_error(err, "clEnqueueReadBuffer best_pos");
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer best_pos");

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
        }

        // Sort particles by fitness
        int *sorted_indices = (int *)malloc(pop_size * sizeof(int));
        for (int i = 0; i < pop_size; i++) sorted_indices[i] = i;
        quicksort_indices(fitness, sorted_indices, 0, pop_size - 1);

        // Reorder positions, velocities, gradients, and fitness
        float *temp_positions = (float *)malloc(pop_size * dim * sizeof(float));
        float *temp_velocities = (float *)malloc(pop_size * dim * sizeof(float));
        float *temp_gradients = (float *)malloc(pop_size * dim * sizeof(float));
        float *temp_fitness = (float *)malloc(pop_size * sizeof(float));

        float *velocities = (float *)malloc(pop_size * dim * sizeof(float));
        float *gradients_reorder = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.position, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer position for sort");
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.velocity, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), velocities, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer velocity for sort");
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.gradient, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), gradients_reorder, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer gradient for sort");

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

        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < dim; j++) {
                temp_positions[i * dim + j] = positions[sorted_indices[i] * dim + j];
                temp_velocities[i * dim + j] = velocities[sorted_indices[i] * dim + j];
                temp_gradients[i * dim + j] = gradients_reorder[sorted_indices[i] * dim + j];
            }
            temp_fitness[i] = fitness[sorted_indices[i]];
        }

        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.position, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), temp_positions, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer position after sort");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.velocity, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), temp_velocities, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer velocity after sort");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.gradient, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), temp_gradients, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer gradient after sort");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), temp_fitness, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer fitness after sort");

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
        free(velocities);
        free(gradients_reorder);
        free(temp_positions);
        free(temp_velocities);
        free(temp_gradients);
        free(temp_fitness);
        free(sorted_indices);
        free(fitness);

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.best_fitness);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Particles: %.3f\n", prof.init_time);
    printf("Update Velocity/Position: %.3f\n", prof.update_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Update Velocity/Position: %.3f\n", prof.update_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Copy best solution to optimizer
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
    EVO_cleanup_context(&ctx, &cl_ctx);
    EVO_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
