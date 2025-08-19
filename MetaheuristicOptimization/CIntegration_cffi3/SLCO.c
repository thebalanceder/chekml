#include "SLCO.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (stub, can be expanded if needed)
char *preprocess_kernel_source(const char *source);

void SLCO_init_cl(SLCOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->init_league_kernel = clCreateKernel(cl_ctx->program, "initialize_league", &err);
    check_cl_error(err, "clCreateKernel init_league");
    cl_ctx->takhsis_kernel = clCreateKernel(cl_ctx->program, "takhsis", &err);
    check_cl_error(err, "clCreateKernel takhsis");
    cl_ctx->winner_main_kernel = clCreateKernel(cl_ctx->program, "winner_function_main", &err);
    check_cl_error(err, "clCreateKernel winner_function_main");
    cl_ctx->winner_reserve_kernel = clCreateKernel(cl_ctx->program, "winner_function_reserve", &err);
    check_cl_error(err, "clCreateKernel winner_function_reserve");
    cl_ctx->loser_kernel = clCreateKernel(cl_ctx->program, "loser_function", &err);
    check_cl_error(err, "clCreateKernel loser_function");
}

void SLCO_cleanup_cl(SLCOCLContext *cl_ctx) {
    if (cl_ctx->loser_kernel) clReleaseKernel(cl_ctx->loser_kernel);
    if (cl_ctx->winner_reserve_kernel) clReleaseKernel(cl_ctx->winner_reserve_kernel);
    if (cl_ctx->winner_main_kernel) clReleaseKernel(cl_ctx->winner_main_kernel);
    if (cl_ctx->takhsis_kernel) clReleaseKernel(cl_ctx->takhsis_kernel);
    if (cl_ctx->init_league_kernel) clReleaseKernel(cl_ctx->init_league_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void SLCO_init_context(SLCOContext *ctx, Optimizer *opt, SLCOCLContext *cl_ctx) {
    cl_int err;
    ctx->best_cost = INFINITY;
    ctx->best_team_idx = 0;

    int n_teams = SLCO_N_TEAMS;
    int n_main = SLCO_N_MAIN_PLAYERS;
    int n_reserve = SLCO_N_RESERVE_PLAYERS;
    int total_players = n_teams * (n_main + n_reserve);
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->positions = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, total_players * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer positions");
    ctx->costs = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, total_players * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer costs");
    ctx->best_position = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_position");
    ctx->bounds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, dim * 2 * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer bounds");
    ctx->random_seeds = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, total_players * sizeof(cl_uint), NULL, &err);
    check_cl_error(err, "clCreateBuffer random_seeds");
    ctx->teams = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, total_players * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer teams");
    ctx->team_total_costs = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, n_teams * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer team_total_costs");

    // Initialize random seeds
    cl_uint *seeds = (cl_uint *)malloc(total_players * sizeof(cl_uint));
    srand(time(NULL));
    for (int i = 0; i < total_players; i++) seeds[i] = rand();
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->random_seeds, CL_TRUE, 0, total_players * sizeof(cl_uint), seeds, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer seeds");
    free(seeds);

    // Initialize bounds
    float *bounds_float = (float *)malloc(dim * 2 * sizeof(float));
    for (int i = 0; i < dim * 2; i++) bounds_float[i] = (float)opt->bounds[i];
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->bounds, CL_TRUE, 0, dim * 2 * sizeof(cl_float), bounds_float, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer bounds");
    free(bounds_float);

    // Initialize team assignments
    cl_int *team_assignments = (cl_int *)malloc(total_players * sizeof(cl_int));
    for (int i = 0; i < total_players; i++) {
        team_assignments[i] = i / (n_main + n_reserve); // Assign to teams
    }
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->teams, CL_TRUE, 0, total_players * sizeof(cl_int), team_assignments, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer teams");
    free(team_assignments);
}

void SLCO_cleanup_context(SLCOContext *ctx, SLCOCLContext *cl_ctx) {
    if (ctx->team_total_costs) clReleaseMemObject(ctx->team_total_costs);
    if (ctx->teams) clReleaseMemObject(ctx->teams);
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->costs) clReleaseMemObject(ctx->costs);
    if (ctx->positions) clReleaseMemObject(ctx->positions);
}

void SLCO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU SLCO optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("SLCO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open SLCO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: SLCO.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read SLCO.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: SLCO.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    SLCOCLContext cl_ctx = {0};
    SLCO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    SLCOContext ctx = {0};
    SLCO_init_context(&ctx, opt, &cl_ctx);

    int n_teams = SLCO_N_TEAMS;
    int n_main = SLCO_N_MAIN_PLAYERS;
    int n_reserve = SLCO_N_RESERVE_PLAYERS;
    int total_players = n_teams * (n_main + n_reserve);
    int dim = opt->dim;
    int func_count = 0;
    int iteration = 0;
    int stagnation_count = 0;
    float prev_best_cost = INFINITY;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * total_players : SLCO_MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_league_time;
        double takhsis_time;
        double winner_main_time;
        double winner_reserve_time;
        double loser_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;
    cl_mem best_cost_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_cost");

    // Initialize league
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_league_kernel, 0, sizeof(cl_mem), &ctx.positions);
    clSetKernelArg(cl_ctx.init_league_kernel, 1, sizeof(cl_mem), &ctx.costs);
    clSetKernelArg(cl_ctx.init_league_kernel, 2, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_league_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_league_kernel, 4, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_league_kernel, 5, sizeof(cl_int), &total_players);
    size_t global_work_size = total_players;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.init_league_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
    check_cl_error(err, "clEnqueueNDRangeKernel init_league");
    clFinish(cl_ctx.queue);

    cl_ulong start, end;
    if (init_event) {
        err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.init_league_time += (end - start) / 1e6;
            }
        }
        clReleaseEvent(init_event);
    }

    // Evaluate initial population on CPU
    cl_event read_event, write_event;
    float *positions = (float *)malloc(total_players * dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, total_players * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer positions");
    float *costs = (float *)malloc(total_players * sizeof(float));
    for (int i = 0; i < total_players; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
        costs[i] = (float)objective_function(pos);
        free(pos);
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, total_players * sizeof(cl_float), costs, 0, NULL, &write_event);
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

    func_count += total_players;
    float best_cost = costs[0];
    int best_idx = 0;
    for (int i = 1; i < total_players; i++) {
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

    // Perform initial takhsis
    cl_event takhsis_event;
    clSetKernelArg(cl_ctx.takhsis_kernel, 0, sizeof(cl_mem), &ctx.positions);
    clSetKernelArg(cl_ctx.takhsis_kernel, 1, sizeof(cl_mem), &ctx.costs);
    clSetKernelArg(cl_ctx.takhsis_kernel, 2, sizeof(cl_mem), &ctx.teams);
    clSetKernelArg(cl_ctx.takhsis_kernel, 3, sizeof(cl_mem), &ctx.team_total_costs);
    clSetKernelArg(cl_ctx.takhsis_kernel, 4, sizeof(cl_int), &n_teams);
    clSetKernelArg(cl_ctx.takhsis_kernel, 5, sizeof(cl_int), &n_main);
    clSetKernelArg(cl_ctx.takhsis_kernel, 6, sizeof(cl_int), &n_reserve);
    clSetKernelArg(cl_ctx.takhsis_kernel, 7, sizeof(cl_int), &dim);
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.takhsis_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &takhsis_event);
    check_cl_error(err, "clEnqueueNDRangeKernel takhsis");
    clFinish(cl_ctx.queue);

    if (takhsis_event) {
        err = clGetEventProfilingInfo(takhsis_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(takhsis_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.takhsis_time += (end - start) / 1e6;
            }
        }
        clReleaseEvent(takhsis_event);
    }

    // Main optimization loop
    while (func_count < max_evals) {
        iteration++;
        printf("Iteration %d\n", iteration);

        float alpha = SLCO_ALPHA_MAX - (SLCO_ALPHA_MAX - SLCO_ALPHA_MIN) * iteration / opt->max_iter;
        float beta = SLCO_BETA_MAX - (SLCO_BETA_MAX - SLCO_BETA_MIN) * iteration / opt->max_iter;

        // Perform competition
        for (int ii = 0; ii < n_teams - 1; ii++) {
            for (int jj = ii + 1; jj < n_teams; jj++) {
                // Determine winner and loser (CPU-based for simplicity)
                float *team_costs = (float *)malloc(n_teams * sizeof(float));
                err = clEnqueueReadBuffer(cl_ctx.queue, ctx.team_total_costs, CL_TRUE, 0, n_teams * sizeof(cl_float), team_costs, 0, NULL, &read_event);
                check_cl_error(err, "clEnqueueReadBuffer team_total_costs");
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

                int winner, loser;
                float p_simple = team_costs[ii] / (team_costs[ii] + team_costs[jj]);
                float r = (float)rand() / RAND_MAX;
                if (r < p_simple) {
                    winner = ii;
                    loser = jj;
                } else {
                    winner = jj;
                    loser = ii;
                }
                free(team_costs);
                ctx.best_team_idx = winner;

                // Read teams buffer for winner main players
                cl_int *teams_host = (cl_int *)malloc(total_players * sizeof(cl_int));
                err = clEnqueueReadBuffer(cl_ctx.queue, ctx.teams, CL_TRUE, 0, total_players * sizeof(cl_int), teams_host, 0, NULL, &read_event);
                check_cl_error(err, "clEnqueueReadBuffer teams for winner main");
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

                // Update winner main players
                cl_event winner_main_event;
                clSetKernelArg(cl_ctx.winner_main_kernel, 0, sizeof(cl_mem), &ctx.positions);
                clSetKernelArg(cl_ctx.winner_main_kernel, 1, sizeof(cl_mem), &ctx.costs);
                clSetKernelArg(cl_ctx.winner_main_kernel, 2, sizeof(cl_mem), &ctx.teams);
                clSetKernelArg(cl_ctx.winner_main_kernel, 3, sizeof(cl_mem), &ctx.bounds);
                clSetKernelArg(cl_ctx.winner_main_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
                clSetKernelArg(cl_ctx.winner_main_kernel, 5, sizeof(cl_float), &alpha);
                clSetKernelArg(cl_ctx.winner_main_kernel, 6, sizeof(cl_float), &beta);
                clSetKernelArg(cl_ctx.winner_main_kernel, 7, sizeof(cl_int), &winner);
                clSetKernelArg(cl_ctx.winner_main_kernel, 8, sizeof(cl_int), &n_teams);
                clSetKernelArg(cl_ctx.winner_main_kernel, 9, sizeof(cl_int), &n_main);
                clSetKernelArg(cl_ctx.winner_main_kernel, 10, sizeof(cl_int), &dim);
                err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.winner_main_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &winner_main_event);
                check_cl_error(err, "clEnqueueNDRangeKernel winner_function_main");
                clFinish(cl_ctx.queue);

                if (winner_main_event) {
                    err = clGetEventProfilingInfo(winner_main_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                    if (err == CL_SUCCESS) {
                        err = clGetEventProfilingInfo(winner_main_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                        if (err == CL_SUCCESS) {
                            prof.winner_main_time += (end - start) / 1e6;
                        }
                    }
                    clReleaseEvent(winner_main_event);
                }

                // Evaluate updated winner main players
                positions = (float *)malloc(total_players * dim * sizeof(float));
                err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, total_players * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
                check_cl_error(err, "clEnqueueReadBuffer positions");
                for (int i = 0; i < total_players; i++) {
                    if (teams_host[i] == winner && i % (n_main + n_reserve) < n_main) {
                        double *pos = (double *)malloc(dim * sizeof(double));
                        for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
                        costs[i] = (float)objective_function(pos);
                        free(pos);
                    }
                }
                err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, total_players * sizeof(cl_float), costs, 0, NULL, &write_event);
                check_cl_error(err, "clEnqueueWriteBuffer costs");
                free(positions);
                func_count += n_main;

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

                // Update winner reserve players
                cl_event winner_reserve_event;
                clSetKernelArg(cl_ctx.winner_reserve_kernel, 0, sizeof(cl_mem), &ctx.positions);
                clSetKernelArg(cl_ctx.winner_reserve_kernel, 1, sizeof(cl_mem), &ctx.costs);
                clSetKernelArg(cl_ctx.winner_reserve_kernel, 2, sizeof(cl_mem), &ctx.teams);
                clSetKernelArg(cl_ctx.winner_reserve_kernel, 3, sizeof(cl_mem), &ctx.bounds);
                clSetKernelArg(cl_ctx.winner_reserve_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
                clSetKernelArg(cl_ctx.winner_reserve_kernel, 5, sizeof(cl_float), &beta);
                clSetKernelArg(cl_ctx.winner_reserve_kernel, 6, sizeof(cl_int), &winner);
                clSetKernelArg(cl_ctx.winner_reserve_kernel, 7, sizeof(cl_int), &n_teams);
                clSetKernelArg(cl_ctx.winner_reserve_kernel, 8, sizeof(cl_int), &n_main);
                clSetKernelArg(cl_ctx.winner_reserve_kernel, 9, sizeof(cl_int), &n_reserve);
                clSetKernelArg(cl_ctx.winner_reserve_kernel, 10, sizeof(cl_int), &dim);
                err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.winner_reserve_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &winner_reserve_event);
                check_cl_error(err, "clEnqueueNDRangeKernel winner_function_reserve");
                clFinish(cl_ctx.queue);

                if (winner_reserve_event) {
                    err = clGetEventProfilingInfo(winner_reserve_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                    if (err == CL_SUCCESS) {
                        err = clGetEventProfilingInfo(winner_reserve_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                        if (err == CL_SUCCESS) {
                            prof.winner_reserve_time += (end - start) / 1e6;
                        }
                    }
                    clReleaseEvent(winner_reserve_event);
                }

                // Evaluate updated winner reserve players
                positions = (float *)malloc(total_players * dim * sizeof(float));
                err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, total_players * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
                check_cl_error(err, "clEnqueueReadBuffer positions");
                for (int i = 0; i < total_players; i++) {
                    if (teams_host[i] == winner && i % (n_main + n_reserve) >= n_main) {
                        double *pos = (double *)malloc(dim * sizeof(double));
                        for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
                        costs[i] = (float)objective_function(pos);
                        free(pos);
                    }
                }
                err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, total_players * sizeof(cl_float), costs, 0, NULL, &write_event);
                check_cl_error(err, "clEnqueueWriteBuffer costs");
                free(positions);
                func_count += n_reserve;

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

                // Free teams_host after winner updates
                free(teams_host);

                // Read teams buffer for loser team
                teams_host = (cl_int *)malloc(total_players * sizeof(cl_int));
                err = clEnqueueReadBuffer(cl_ctx.queue, ctx.teams, CL_TRUE, 0, total_players * sizeof(cl_int), teams_host, 0, NULL, &read_event);
                check_cl_error(err, "clEnqueueReadBuffer teams for loser");
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

                // Update loser team
                cl_event loser_event;
                clSetKernelArg(cl_ctx.loser_kernel, 0, sizeof(cl_mem), &ctx.positions);
                clSetKernelArg(cl_ctx.loser_kernel, 1, sizeof(cl_mem), &ctx.costs);
                clSetKernelArg(cl_ctx.loser_kernel, 2, sizeof(cl_mem), &ctx.teams);
                clSetKernelArg(cl_ctx.loser_kernel, 3, sizeof(cl_mem), &ctx.bounds);
                clSetKernelArg(cl_ctx.loser_kernel, 4, sizeof(cl_mem), &ctx.random_seeds);
                clSetKernelArg(cl_ctx.loser_kernel, 5, sizeof(cl_float), &alpha);
                clSetKernelArg(cl_ctx.loser_kernel, 6, sizeof(cl_int), &loser);
                clSetKernelArg(cl_ctx.loser_kernel, 7, sizeof(cl_int), &n_teams);
                clSetKernelArg(cl_ctx.loser_kernel, 8, sizeof(cl_int), &n_main);
                clSetKernelArg(cl_ctx.loser_kernel, 9, sizeof(cl_int), &n_reserve);
                clSetKernelArg(cl_ctx.loser_kernel, 10, sizeof(cl_int), &dim);
                clSetKernelArg(cl_ctx.loser_kernel, 11, sizeof(cl_float), &iteration);
                clSetKernelArg(cl_ctx.loser_kernel, 12, sizeof(cl_int), &opt->max_iter);
                err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.loser_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &loser_event);
                check_cl_error(err, "clEnqueueNDRangeKernel loser_function");
                clFinish(cl_ctx.queue);

                if (loser_event) {
                    err = clGetEventProfilingInfo(loser_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                    if (err == CL_SUCCESS) {
                        err = clGetEventProfilingInfo(loser_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                        if (err == CL_SUCCESS) {
                            prof.loser_time += (end - start) / 1e6;
                        }
                    }
                    clReleaseEvent(loser_event);
                }

                // Evaluate updated loser team
                positions = (float *)malloc(total_players * dim * sizeof(float));
                err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, total_players * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
                check_cl_error(err, "clEnqueueReadBuffer positions");
                for (int i = 0; i < total_players; i++) {
                    if (teams_host[i] == loser) {
                        double *pos = (double *)malloc(dim * sizeof(double));
                        for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
                        costs[i] = (float)objective_function(pos);
                        free(pos);
                    }
                }
                err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, total_players * sizeof(cl_float), costs, 0, NULL, &write_event);
                check_cl_error(err, "clEnqueueWriteBuffer costs");
                free(positions);
                func_count += (n_main + n_reserve);

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

                // Free teams_host after loser updates
                free(teams_host);

                // Update team total costs
                err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.takhsis_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &takhsis_event);
                check_cl_error(err, "clEnqueueNDRangeKernel takhsis");
                clFinish(cl_ctx.queue);

                if (takhsis_event) {
                    err = clGetEventProfilingInfo(takhsis_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
                    if (err == CL_SUCCESS) {
                        err = clGetEventProfilingInfo(takhsis_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                        if (err == CL_SUCCESS) {
                            prof.takhsis_time += (end - start) / 1e6;
                        }
                    }
                    clReleaseEvent(takhsis_event);
                }
            }
        }

        // Update best solution
        float new_best = costs[0];
        int best_idx = 0;
        for (int i = 1; i < total_players; i++) {
            if (costs[i] < new_best) {
                new_best = costs[i];
                best_idx = i;
            }
        }
        if (new_best < ctx.best_cost) {
            ctx.best_cost = new_best;
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
            if (fabs(prev_best_cost - new_best) < SLCO_CONVERGENCE_TOL) {
                stagnation_count++;
            } else {
                stagnation_count = 0;
            }
            prev_best_cost = new_best;
            if (stagnation_count >= SLCO_STAGNATION_THRESHOLD) {
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
    printf("Initialize League: %.3f\n", prof.init_league_time);
    printf("Takhsis: %.3f\n", prof.takhsis_time);
    printf("Winner Main Update: %.3f\n", prof.winner_main_time);
    printf("Winner Reserve Update: %.3f\n", prof.winner_reserve_time);
    printf("Loser Update: %.3f\n", prof.loser_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Takhsis: %.3f\n", prof.takhsis_time / prof.count);
        printf("Winner Main Update: %.3f\n", prof.winner_main_time / prof.count);
        printf("Winner Reserve Update: %.3f\n", prof.winner_reserve_time / prof.count);
        printf("Loser Update: %.3f\n", prof.loser_time / prof.count);
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
    SLCO_cleanup_context(&ctx, &cl_ctx);
    SLCO_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
