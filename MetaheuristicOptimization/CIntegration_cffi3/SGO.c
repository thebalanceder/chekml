#include "SGO.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (stub)
char *preprocess_kernel_source(const char *source);

void SGO_init_cl(SGOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->init_players_kernel = clCreateKernel(cl_ctx->program, "initialize_players", &err);
    check_cl_error(err, "clCreateKernel initialize_players");
    cl_ctx->divide_teams_kernel = clCreateKernel(cl_ctx->program, "divide_teams", &err);
    check_cl_error(err, "clCreateKernel divide_teams");
    cl_ctx->simulate_fight_kernel = clCreateKernel(cl_ctx->program, "simulate_fight", &err);
    check_cl_error(err, "clCreateKernel simulate_fight");
    cl_ctx->determine_winners_kernel = clCreateKernel(cl_ctx->program, "determine_winners", &err);
    check_cl_error(err, "clCreateKernel determine_winners");
    cl_ctx->update_positions_kernel = clCreateKernel(cl_ctx->program, "update_positions", &err);
    check_cl_error(err, "clCreateKernel update_positions");
}

void SGO_cleanup_cl(SGOCLContext *cl_ctx) {
    if (cl_ctx->update_positions_kernel) clReleaseKernel(cl_ctx->update_positions_kernel);
    if (cl_ctx->determine_winners_kernel) clReleaseKernel(cl_ctx->determine_winners_kernel);
    if (cl_ctx->simulate_fight_kernel) clReleaseKernel(cl_ctx->simulate_fight_kernel);
    if (cl_ctx->divide_teams_kernel) clReleaseKernel(cl_ctx->divide_teams_kernel);
    if (cl_ctx->init_players_kernel) clReleaseKernel(cl_ctx->init_players_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void SGO_init_context(SGOContext *ctx, Optimizer *opt, SGOCLContext *cl_ctx) {
    cl_int err;
    ctx->best_cost = INFINITY;
    ctx->offensive_size = (int)(opt->population_size * SGO_ATTACK_RATE);

    int population_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->positions = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer positions");
    ctx->costs = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer costs");
    ctx->best_position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_position");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->team_assignments = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer team_assignments");
    ctx->winners = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, population_size * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer winners");

    // Initialize random seeds
    cl_uint *seeds = (cl_uint *)malloc(population_size * sizeof(cl_uint));
    srand(time(NULL));
    for (int i = 0; i < population_size; i++) seeds[i] = rand();
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->random_seeds, CL_TRUE, 0, population_size * sizeof(cl_uint), seeds, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer seeds");
    free(seeds);

    // Initialize bounds
    float *bounds_float = (float *)malloc(dim * 2 * sizeof(float));
    for (int i = 0; i < dim * 2; i++) bounds_float[i] = (float)opt->bounds[i];
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->bounds, CL_TRUE, 0, dim * 2 * sizeof(cl_float), bounds_float, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer bounds");
    free(bounds_float);
}

void SGO_cleanup_context(SGOContext *ctx, SGOCLContext *cl_ctx) {
    if (ctx->winners) clReleaseMemObject(ctx->winners);
    if (ctx->team_assignments) clReleaseMemObject(ctx->team_assignments);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->costs) clReleaseMemObject(ctx->costs);
    if (ctx->positions) clReleaseMemObject(ctx->positions);
}

void SGO_optimize(Optimizer *opt, ObjectiveFunction objective_function) {
    printf("Starting GPU SGO optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("SGO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open SGO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: SGO.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read SGO.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: SGO.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    SGOCLContext cl_ctx = {0};
    SGO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    SGOContext ctx = {0};
    SGO_init_context(&ctx, opt, &cl_ctx);

    int population_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;
    int stagnation_count = 0;
    float prev_best_cost = INFINITY;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * population_size : SGO_MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_players_time;
        double divide_teams_time;
        double simulate_fight_time;
        double determine_winners_time;
        double update_positions_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;
    cl_mem best_cost_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_cost");

    // Initialize players
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_players_kernel, 0, sizeof(cl_mem), &ctx.positions);
    clSetKernelArg(cl_ctx.init_players_kernel, 1, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_players_kernel, 2, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_players_kernel, 3, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_players_kernel, 4, sizeof(cl_int), &population_size);
    size_t global_work_size = population_size;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.init_players_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
    check_cl_error(err, "clEnqueueNDRangeKernel init_players");
    clFinish(cl_ctx.queue);

    cl_ulong start, end;
    if (init_event) {
        err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.init_players_time += (end - start) / 1e6;
            }
        }
        clReleaseEvent(init_event);
    }

    // Evaluate initial population on CPU
    cl_event read_event, write_event;
    float *positions = (float *)malloc(population_size * dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer positions");
    float *costs = (float *)malloc(population_size * sizeof(float));
    for (int i = 0; i < population_size; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
        costs[i] = (float)objective_function(pos);
        free(pos);
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, population_size * sizeof(cl_float), costs, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer costs");
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

    func_count += population_size;
    float best_cost = costs[0];
    int best_idx = 0;
    for (int i = 1; i < population_size; i++) {
        if (costs[i] < best_cost) {
            best_cost = costs[i];
            best_idx = i;
        }
    }
    ctx.best_cost = best_cost;

    positions = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
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

    err = clEnqueueWriteBuffer(cl_ctx.queue, best_cost_buf, CL_TRUE, 0, sizeof(cl_float), &best_cost, 0, NULL, &write_event);
    check_cl_error(err, "clEnqueueWriteBuffer best_cost");

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

        // Divide teams
        cl_event divide_event;
        clSetKernelArg(cl_ctx.divide_teams_kernel, 0, sizeof(cl_mem), &ctx.team_assignments);
        clSetKernelArg(cl_ctx.divide_teams_kernel, 1, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.divide_teams_kernel, 2, sizeof(cl_int), &population_size);
        clSetKernelArg(cl_ctx.divide_teams_kernel, 3, sizeof(cl_int), &ctx.offensive_size);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.divide_teams_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &divide_event);
        check_cl_error(err, "clEnqueueNDRangeKernel divide_teams");
        clFinish(cl_ctx.queue);

        if (divide_event) {
            err = clGetEventProfilingInfo(divide_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(divide_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.divide_teams_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(divide_event);
        }

        // Simulate fights
        cl_event fight_event;
        int fight_count = ctx.offensive_size < (population_size - ctx.offensive_size) ? ctx.offensive_size : (population_size - ctx.offensive_size);
        size_t fight_work_size = fight_count;
        clSetKernelArg(cl_ctx.simulate_fight_kernel, 0, sizeof(cl_mem), &ctx.positions);
        clSetKernelArg(cl_ctx.simulate_fight_kernel, 1, sizeof(cl_mem), &ctx.team_assignments);
        clSetKernelArg(cl_ctx.simulate_fight_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.simulate_fight_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.simulate_fight_kernel, 4, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.simulate_fight_kernel, 5, sizeof(cl_int), &population_size);
        clSetKernelArg(cl_ctx.simulate_fight_kernel, 6, sizeof(cl_int), &ctx.offensive_size);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.simulate_fight_kernel, 1, NULL, &fight_work_size, NULL, 0, NULL, &fight_event);
        check_cl_error(err, "clEnqueueNDRangeKernel simulate_fight");
        clFinish(cl_ctx.queue);

        if (fight_event) {
            err = clGetEventProfilingInfo(fight_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(fight_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            }
        }

        // Evaluate updated population
        positions = (float *)malloc(population_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer positions");
        for (int i = 0; i < population_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            costs[i] = (float)objective_function(pos);
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, population_size * sizeof(cl_float), costs, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer costs");
        free(positions);
        func_count += population_size;

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

        // Determine winners
        cl_event winners_event;
        clSetKernelArg(cl_ctx.determine_winners_kernel, 0, sizeof(cl_mem), &ctx.costs);
        clSetKernelArg(cl_ctx.determine_winners_kernel, 1, sizeof(cl_mem), &ctx.team_assignments);
        clSetKernelArg(cl_ctx.determine_winners_kernel, 2, sizeof(cl_mem), &ctx.winners);
        clSetKernelArg(cl_ctx.determine_winners_kernel, 3, sizeof(cl_int), &population_size);
        clSetKernelArg(cl_ctx.determine_winners_kernel, 4, sizeof(cl_int), &ctx.offensive_size);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.determine_winners_kernel, 1, NULL, &fight_work_size, NULL, 0, NULL, &winners_event);
        check_cl_error(err, "clEnqueueNDRangeKernel determine_winners");
        clFinish(cl_ctx.queue);

        if (winners_event) {
            err = clGetEventProfilingInfo(winners_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(winners_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.determine_winners_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(winners_event);
        }

        // Update positions
        cl_event update_event;
        clSetKernelArg(cl_ctx.update_positions_kernel, 0, sizeof(cl_mem), &ctx.positions);
        clSetKernelArg(cl_ctx.update_positions_kernel, 1, sizeof(cl_mem), &ctx.best_position);
        clSetKernelArg(cl_ctx.update_positions_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.update_positions_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.update_positions_kernel, 4, sizeof(cl_mem), &ctx.winners);
        clSetKernelArg(cl_ctx.update_positions_kernel, 5, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.update_positions_kernel, 6, sizeof(cl_int), &population_size);
        clSetKernelArg(cl_ctx.update_positions_kernel, 7, sizeof(cl_int), &fight_count);
        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_positions_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &update_event);
        check_cl_error(err, "clEnqueueNDRangeKernel update_positions");
        clFinish(cl_ctx.queue);

        if (update_event) {
            err = clGetEventProfilingInfo(update_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(update_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    prof.update_positions_time += (end - start) / 1e6;
                }
            }
            clReleaseEvent(update_event);
        }

        // Evaluate updated population
        positions = (float *)malloc(population_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, population_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer positions");
        for (int i = 0; i < population_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            costs[i] = (float)objective_function(pos);
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, population_size * sizeof(cl_float), costs, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer costs");
        free(positions);
        func_count += population_size;

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
        float new_best = costs[0];
        int new_best_idx = 0;
        for (int i = 1; i < population_size; i++) {
            if (costs[i] < new_best) {
                new_best = costs[i];
                new_best_idx = i;
            }
        }
        if (new_best < ctx.best_cost) {
            ctx.best_cost = new_best;
            best_idx = new_best_idx;
            positions = (float *)malloc(dim * sizeof(float));
            err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, &read_event);
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

            err = clEnqueueWriteBuffer(cl_ctx.queue, best_cost_buf, CL_TRUE, 0, sizeof(cl_float), &new_best, 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer best_cost");

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

            // Check for convergence
            if (fabs(prev_best_cost - new_best) < SGO_CONVERGENCE_TOL) {
                stagnation_count++;
            } else {
                stagnation_count = 0;
            }
            prev_best_cost = new_best;
            if (stagnation_count >= SGO_STAGNATION_THRESHOLD) {
                printf("Convergence reached, stopping early\n");
                break;
            }
        }

        prof.count++;
        printf("Iteration %d: Function Evaluations = %d, Best Cost = %f\n", iteration, func_count, ctx.best_cost);
    }

    // Free costs after the loop
    free(costs);

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Players: %.3f\n", prof.init_players_time);
    printf("Divide Teams: %.3f\n", prof.divide_teams_time);
    printf("Simulate Fight: %.3f\n", prof.simulate_fight_time);
    printf("Determine Winners: %.3f\n", prof.determine_winners_time);
    printf("Update Positions: %.3f\n", prof.update_positions_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Divide Teams: %.3f\n", prof.divide_teams_time / prof.count);
        printf("Simulate Fight: %.3f\n", prof.simulate_fight_time / prof.count);
        printf("Determine Winners: %.3f\n", prof.determine_winners_time / prof.count);
        printf("Update Positions: %.3f\n", prof.update_positions_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Copy best solution to optimizer
    float *best_pos = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer final best_pos");
    opt->best_solution.fitness = ctx.best_cost;
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
    clReleaseMemObject(best_cost_buf);
    SGO_cleanup_context(&ctx, &cl_ctx);
    SGO_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
