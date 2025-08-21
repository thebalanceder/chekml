#include "ICA.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>

// Error checking utility
void check_cl_error(cl_int err, const char *msg);

// Preprocess kernel source (stub, as in FirefA)
char *preprocess_kernel_source(const char *source);

void ICA_init_cl(ICACLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
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
    cl_ctx->init_countries_kernel = clCreateKernel(cl_ctx->program, "initialize_countries", &err);
    check_cl_error(err, "clCreateKernel init_countries");
    cl_ctx->assimilate_kernel = clCreateKernel(cl_ctx->program, "assimilate_colonies", &err);
    check_cl_error(err, "clCreateKernel assimilate_colonies");
}

void ICA_cleanup_cl(ICACLContext *cl_ctx) {
    if (cl_ctx->assimilate_kernel) clReleaseKernel(cl_ctx->assimilate_kernel);
    if (cl_ctx->init_countries_kernel) clReleaseKernel(cl_ctx->init_countries_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    if (cl_ctx->queue && cl_ctx->owns_queue) clReleaseCommandQueue(cl_ctx->queue);
}

void ICA_init_context(ICAContext *ctx, Optimizer *opt, ICACLContext *cl_ctx) {
    cl_int err;
    ctx->best_cost = INFINITY;

    int pop_size = opt->population_size;
    int dim = opt->dim;

    // Allocate OpenCL buffers
    ctx->positions = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer positions");
    ctx->costs = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer costs");
    ctx->empire_indices = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer empire_indices");
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

void ICA_cleanup_context(ICAContext *ctx, ICACLContext *cl_ctx) {
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->empire_indices) clReleaseMemObject(ctx->empire_indices);
    if (ctx->costs) clReleaseMemObject(ctx->costs);
    if (ctx->positions) clReleaseMemObject(ctx->positions);
}

typedef struct {
    int *imperialist_indices; // Indices of imperialist countries
    int num_empires;          // Number of empires
    int *num_colonies;        // Number of colonies per empire
    int *colony_indices;      // Indices of colonies per empire
} EmpireAssignment;

void assign_empires(Optimizer *opt, ICAContext *ctx, ICACLContext *cl_ctx, double (*objective_function)(double *), EmpireAssignment *ea) {
    int pop_size = opt->population_size;
    float *costs = (float *)malloc(pop_size * sizeof(float));
    int *indices = (int *)malloc(pop_size * sizeof(int));
    cl_int err;

    // Read positions and evaluate costs
    float *positions = (float *)malloc(pop_size * opt->dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx->queue, ctx->positions, CL_TRUE, 0, pop_size * opt->dim * sizeof(cl_float), positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueReadBuffer positions");
    for (int i = 0; i < pop_size; i++) {
        double *pos = (double *)malloc(opt->dim * sizeof(double));
        for (int j = 0; j < opt->dim; j++) pos[j] = positions[i * opt->dim + j];
        costs[i] = (float)objective_function(pos);
        indices[i] = i;
        free(pos);
    }
    free(positions);

    // Sort by cost
    for (int i = 0; i < pop_size - 1; i++) {
        for (int j = i + 1; j < pop_size; j++) {
            if (costs[indices[i]] > costs[indices[j]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }

    // Assign imperialists and colonies
    int effective_imperialists = ICA_NUM_IMPERIALISTS;
    int num_colonies_total = pop_size - effective_imperialists;
    if (num_colonies_total < effective_imperialists) {
        effective_imperialists = pop_size / 2;
        if (effective_imperialists < 1) effective_imperialists = 1;
        num_colonies_total = pop_size - effective_imperialists;
    }
    ea->num_empires = effective_imperialists;
    ea->imperialist_indices = (int *)malloc(effective_imperialists * sizeof(int));
    ea->num_colonies = (int *)calloc(effective_imperialists, sizeof(int));
    ea->colony_indices = (int *)malloc(num_colonies_total * sizeof(int));

    // Compute normalized power
    float max_cost = costs[indices[0]];
    for (int i = 1; i < effective_imperialists; i++) {
        if (costs[indices[i]] > max_cost) max_cost = costs[indices[i]];
    }
    float *power = (float *)malloc(effective_imperialists * sizeof(float));
    float power_sum = 0.0f;
    for (int i = 0; i < effective_imperialists; i++) {
        power[i] = max_cost - costs[indices[i]];
        power_sum += power[i];
    }

    // Assign colonies
    int remaining = num_colonies_total;
    for (int i = 0; i < effective_imperialists - 1; i++) {
        ea->imperialist_indices[i] = indices[i];
        ea->num_colonies[i] = (int)(power[i] / power_sum * num_colonies_total);
        if (ea->num_colonies[i] > remaining) ea->num_colonies[i] = remaining;
        if (ea->num_colonies[i] < 0) ea->num_colonies[i] = 0;
        remaining -= ea->num_colonies[i];
    }
    ea->imperialist_indices[effective_imperialists - 1] = indices[effective_imperialists - 1];
    ea->num_colonies[effective_imperialists - 1] = remaining;

    int colony_idx = 0;
    for (int i = effective_imperialists; i < pop_size; i++) {
        ea->colony_indices[colony_idx++] = indices[i];
    }

    // Shuffle colony indices
    for (int i = num_colonies_total - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = ea->colony_indices[i];
        ea->colony_indices[i] = ea->colony_indices[j];
        ea->colony_indices[j] = temp;
    }

    // Assign empire indices
    int *empire_indices = (int *)calloc(pop_size, sizeof(int));
    colony_idx = 0;
    for (int i = 0; i < effective_imperialists; i++) {
        empire_indices[ea->imperialist_indices[i]] = i;
        for (int j = 0; j < ea->num_colonies[i]; j++) {
            if (colony_idx < num_colonies_total) {
                empire_indices[ea->colony_indices[colony_idx]] = i;
                colony_idx++;
            }
        }
    }
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->empire_indices, CL_TRUE, 0, pop_size * sizeof(cl_int), empire_indices, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer empire_indices");

    // Update best solution
    ctx->best_cost = costs[indices[0]];
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->best_position, CL_TRUE, 0, opt->dim * sizeof(cl_float), &positions[indices[0] * opt->dim], 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer best_position");

    free(costs);
    free(indices);
    free(power);
    free(empire_indices);
}

void ICA_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU Imperialist Competitive Algorithm optimization...\n");

    // Validate inputs
    if (!opt || !opt->context || !opt->queue || !opt->device || !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("ICA.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open ICA.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: ICA.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read ICA.cl completely (%zu of %zu bytes)\n", read_size, source_size);
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
        fprintf(stderr, "Error: ICA.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    // Initialize OpenCL context
    ICACLContext cl_ctx = {0};
    ICA_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    // Initialize algorithm context
    ICAContext ctx = {0};
    ICA_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int func_count = 0;
    int decade = 0;
    float revolution_rate = ICA_REVOLUTION_RATE;

    // Compute max evaluations
    int max_evals = opt->max_iter > 0 ? opt->max_iter * pop_size : MAX_EVALS_DEFAULT;
    printf("Maximum function evaluations: %d\n", max_evals);

    // Profiling data
    typedef struct {
        double init_countries_time;
        double assimilate_time;
        double read_time;
        double write_time;
        int count;
    } ProfilingData;
    ProfilingData prof = {0};

    cl_int err;
    cl_mem best_cost_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_cost");

    // Initialize countries
    cl_event init_event;
    clSetKernelArg(cl_ctx.init_countries_kernel, 0, sizeof(cl_mem), &ctx.positions);
    clSetKernelArg(cl_ctx.init_countries_kernel, 1, sizeof(cl_mem), &ctx.costs);
    clSetKernelArg(cl_ctx.init_countries_kernel, 2, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_countries_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_countries_kernel, 4, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_countries_kernel, 5, sizeof(cl_int), &pop_size);

    size_t global_work_size = pop_size;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.init_countries_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &init_event);
    check_cl_error(err, "clEnqueueNDRangeKernel init_countries");
    clFinish(cl_ctx.queue);

    cl_ulong start, end;
    if (init_event) {
        err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(init_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.init_countries_time += (end - start) / 1e6; // Convert ns to ms
            }
        }
        clReleaseEvent(init_event);
    }

    // Assign empires and evaluate initial costs
    EmpireAssignment ea = {0};
    assign_empires(opt, &ctx, &cl_ctx, objective_function, &ea);
    func_count += pop_size;

    // Main optimization loop
    while (func_count < max_evals && ea.num_empires > 1) {
        decade++;
        revolution_rate *= ICA_DAMP_RATIO;
        printf("Decade %d: Empires = %d\n", decade, ea.num_empires);

        // Assimilate colonies
        cl_event assimilate_event;
        cl_float assim_coeff = ICA_ASSIMILATION_COEFF;
        clSetKernelArg(cl_ctx.assimilate_kernel, 0, sizeof(cl_mem), &ctx.positions);
        clSetKernelArg(cl_ctx.assimilate_kernel, 1, sizeof(cl_mem), &ctx.empire_indices);
        clSetKernelArg(cl_ctx.assimilate_kernel, 2, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.assimilate_kernel, 3, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.assimilate_kernel, 4, sizeof(cl_float), &assim_coeff);
        clSetKernelArg(cl_ctx.assimilate_kernel, 5, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.assimilate_kernel, 6, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.assimilate_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &assimilate_event);
        check_cl_error(err, "clEnqueueNDRangeKernel assimilate_colonies");
        clFinish(cl_ctx.queue);

        double assimilate_time = 0.0;
        if (assimilate_event) {
            err = clGetEventProfilingInfo(assimilate_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            if (err == CL_SUCCESS) {
                err = clGetEventProfilingInfo(assimilate_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
                if (err == CL_SUCCESS) {
                    assimilate_time = (end - start) / 1e6;
                    prof.assimilate_time += assimilate_time;
                }
            }
            clReleaseEvent(assimilate_event);
        }

        // Evaluate population and update empires
        float *positions = (float *)malloc(pop_size * dim * sizeof(float));
        float *costs = (float *)malloc(pop_size * sizeof(float));
        cl_event read_event, write_event;
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.positions, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, &read_event);
        check_cl_error(err, "clEnqueueReadBuffer positions");
        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            costs[i] = (float)objective_function(pos);
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.costs, CL_TRUE, 0, pop_size * sizeof(cl_float), costs, 0, NULL, &write_event);
        check_cl_error(err, "clEnqueueWriteBuffer costs");

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

        // Update best solution and reassign empires
        float new_best = costs[0];
        int best_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (costs[i] < new_best) {
                new_best = costs[i];
                best_idx = i;
            }
        }
        if (new_best < ctx.best_cost) {
            ctx.best_cost = new_best;
            err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), &positions[best_idx * dim], 0, NULL, &write_event);
            check_cl_error(err, "clEnqueueWriteBuffer best_position");
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
        free(costs);
        free(positions);

        // Reassign empires
        free(ea.imperialist_indices);
        free(ea.num_colonies);
        free(ea.colony_indices);
        assign_empires(opt, &ctx, &cl_ctx, objective_function, &ea);
        func_count += pop_size;

        prof.count++;
        printf("Decade %d: Function Evaluations = %d, Best Cost = %f\n", decade, func_count, ctx.best_cost);
        printf("Profiling (ms): Assimilate Colonies = %.3f\n", assimilate_time);
    }

    // Print profiling summary
    printf("\nProfiling Summary (Total ms):\n");
    printf("Initialize Countries: %.3f\n", prof.init_countries_time);
    printf("Assimilate Colonies: %.3f\n", prof.assimilate_time);
    printf("Data Reads: %.3f\n", prof.read_time);
    printf("Data Writes: %.3f\n", prof.write_time);
    if (prof.count > 0) {
        printf("Average per Iteration (ms):\n");
        printf("Assimilate Colonies: %.3f\n", prof.assimilate_time / prof.count);
        printf("Data Reads: %.3f\n", prof.read_time / prof.count);
        printf("Data Writes: %.3f\n", prof.write_time / prof.count);
    }

    // Copy best solution to optimizer
    float *best_pos = (float *)malloc(dim * sizeof(float));
    cl_event read_event; // Declare read_event
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, &read_event);
    check_cl_error(err, "clEnqueueReadBuffer final best_pos");
    opt->best_solution.fitness = ctx.best_cost;
    for (int i = 0; i < dim; i++) opt->best_solution.position[i] = best_pos[i];
    free(best_pos);

    if (read_event) {
        cl_ulong start, end; // Declare start and end for this scope
        err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        if (err == CL_SUCCESS) {
            err = clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            if (err == CL_SUCCESS) {
                prof.read_time += (end - start) / 1e6; // Convert ns to ms
            } else {
                fprintf(stderr, "Warning: clGetEventProfilingInfo final read end failed: %d\n", err);
            }
        } else {
            fprintf(stderr, "Warning: clGetEventProfilingInfo final read start failed: %d\n", err);
        }
        clReleaseEvent(read_event);
    }

    // Cleanup
    free(ea.imperialist_indices);
    free(ea.num_colonies);
    free(ea.colony_indices);
    clReleaseMemObject(best_cost_buf);
    ICA_cleanup_context(&ctx, &cl_ctx);
    ICA_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
