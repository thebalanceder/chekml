#include "FlyFO.h"
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <stdlib.h>

void check_cl_error(cl_int err, const char *msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s: Error code %d\n", msg, err);
        exit(1);
    }
}

// Preprocess kernel source to fix rand_float address space
char *preprocess_kernel_source(const char *source) {
    // Estimate output size (add space for modifications)
    size_t source_len = strlen(source);
    size_t out_size = source_len + 1024;
    char *out = (char *)malloc(out_size);
    if (!out) {
        fprintf(stderr, "Error: Memory allocation failed for preprocessed kernel source\n");
        exit(1);
    }
    out[0] = '\0';

    // Replace "inline float rand_float(uint *seed)" with "inline float rand_float(__global uint *seed)"
    const char *rand_float_def = "inline float rand_float(uint *seed)";
    const char *new_rand_float_def = "inline float rand_float(__global uint *seed)";
    char *pos = (char *)source;
    char *last_pos = (char *)source;
    size_t remaining = out_size - 1;

    while ((pos = strstr(pos, rand_float_def)) != NULL) {
        // Copy up to the match
        size_t len = pos - last_pos;
        if (len >= remaining) {
            fprintf(stderr, "Error: Preprocessed kernel source buffer too small\n");
            free(out);
            exit(1);
        }
        strncat(out, last_pos, len);
        remaining -= len;

        // Append the new definition
        size_t new_len = strlen(new_rand_float_def);
        if (new_len >= remaining) {
            fprintf(stderr, "Error: Preprocessed kernel source buffer too small\n");
            free(out);
            exit(1);
        }
        strcat(out, new_rand_float_def);
        remaining -= new_len;

        // Move past the matched string
        pos += strlen(rand_float_def);
        last_pos = pos;
    }

    // Copy the rest of the source
    if (strlen(last_pos) >= remaining) {
        fprintf(stderr, "Error: Preprocessed kernel source buffer too small\n");
        free(out);
        exit(1);
    }
    strcat(out, last_pos);

    return out;
}

void FlyFO_init_cl(FlyFOCLContext *cl_ctx, const char *kernel_source, Optimizer *opt) {
    cl_int err;

    // Validate input pointers
    if (!opt || !opt->context || !opt->queue || !opt->device) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Validate kernel source
    if (!kernel_source || strlen(kernel_source) == 0) {
        fprintf(stderr, "Error: Kernel source is empty or null\n");
        exit(1);
    }

    // Preprocess kernel source to fix address space
    char *processed_source = preprocess_kernel_source(kernel_source);

    // Reuse OpenCL context, device, and queue from Optimizer
    cl_ctx->context = opt->context;
    cl_ctx->device = opt->device;
    cl_ctx->queue = opt->queue;

    cl_ctx->program = clCreateProgramWithSource(cl_ctx->context, 1, (const char **)&processed_source, NULL, &err);
    check_cl_error(err, "clCreateProgramWithSource");

    // Build program with explicit options for compatibility
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

    cl_ctx->init_pop_kernel = clCreateKernel(cl_ctx->program, "initialize_population", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating init_pop kernel: %d\n", err);
        clReleaseProgram(cl_ctx->program);
        exit(1);
    }
    cl_ctx->fuzzy_tuning_kernel = clCreateKernel(cl_ctx->program, "fuzzy_self_tuning", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating fuzzy_tuning kernel: %d\n", err);
        clReleaseKernel(cl_ctx->init_pop_kernel);
        clReleaseProgram(cl_ctx->program);
        exit(1);
    }
    cl_ctx->update_position_kernel = clCreateKernel(cl_ctx->program, "update_position", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating update_position kernel: %d\n", err);
        clReleaseKernel(cl_ctx->init_pop_kernel);
        clReleaseKernel(cl_ctx->fuzzy_tuning_kernel);
        clReleaseProgram(cl_ctx->program);
        exit(1);
    }
    cl_ctx->crossover_kernel = clCreateKernel(cl_ctx->program, "crossover", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating crossover kernel: %d\n", err);
        clReleaseKernel(cl_ctx->init_pop_kernel);
        clReleaseKernel(cl_ctx->fuzzy_tuning_kernel);
        clReleaseKernel(cl_ctx->update_position_kernel);
        clReleaseProgram(cl_ctx->program);
        exit(1);
    }
    cl_ctx->suffocation_kernel = clCreateKernel(cl_ctx->program, "suffocation", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating suffocation kernel: %d\n", err);
        clReleaseKernel(cl_ctx->init_pop_kernel);
        clReleaseKernel(cl_ctx->fuzzy_tuning_kernel);
        clReleaseKernel(cl_ctx->update_position_kernel);
        clReleaseKernel(cl_ctx->crossover_kernel);
        clReleaseProgram(cl_ctx->program);
        exit(1);
    }
}

void FlyFO_cleanup_cl(FlyFOCLContext *cl_ctx) {
    if (cl_ctx->suffocation_kernel) clReleaseKernel(cl_ctx->suffocation_kernel);
    if (cl_ctx->crossover_kernel) clReleaseKernel(cl_ctx->crossover_kernel);
    if (cl_ctx->update_position_kernel) clReleaseKernel(cl_ctx->update_position_kernel);
    if (cl_ctx->fuzzy_tuning_kernel) clReleaseKernel(cl_ctx->fuzzy_tuning_kernel);
    if (cl_ctx->init_pop_kernel) clReleaseKernel(cl_ctx->init_pop_kernel);
    if (cl_ctx->program) clReleaseProgram(cl_ctx->program);
    // Do not release context, queue, or device, as they are managed by Optimizer
}

void FlyFO_init_context(FlyFOContext *ctx, Optimizer *opt, FlyFOCLContext *cl_ctx) {
    cl_int err;
    ctx->best_fitness = INFINITY;
    ctx->worst_fitness = -INFINITY;
    ctx->survival_count = 0;

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int surv_list_size = (int)(pop_size * SURVIVAL_LIST_RATIO);

    ctx->population = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer population");
    ctx->fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer fitness");
    ctx->past_fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer past_fitness");
    ctx->survival_list = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, surv_list_size * dim * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer survival_list");
    ctx->survival_fitness = clCreateBuffer(cl_ctx->context, CL_MEM_READ_WRITE, surv_list_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer survival_fitness");
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

    // Write bounds
    float *bounds_float = (float *)malloc(dim * 2 * sizeof(float));
    for (int i = 0; i < dim * 2; i++) bounds_float[i] = (float)opt->bounds[i];
    err = clEnqueueWriteBuffer(cl_ctx->queue, ctx->bounds, CL_TRUE, 0, dim * 2 * sizeof(cl_float), bounds_float, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer bounds");
    free(bounds_float);
}

void FlyFO_cleanup_context(FlyFOContext *ctx, FlyFOCLContext *cl_ctx) {
    if (ctx->random_seeds) clReleaseMemObject(ctx->random_seeds);
    if (ctx->bounds) clReleaseMemObject(ctx->bounds);
    if (ctx->best_position) clReleaseMemObject(ctx->best_position);
    if (ctx->survival_fitness) clReleaseMemObject(ctx->survival_fitness);
    if (ctx->survival_list) clReleaseMemObject(ctx->survival_list);
    if (ctx->past_fitness) clReleaseMemObject(ctx->past_fitness);
    if (ctx->fitness) clReleaseMemObject(ctx->fitness);
    if (ctx->population) clReleaseMemObject(ctx->population);
}

void FlyFO_optimize(Optimizer *opt, double (*objective_function)(double *)) {
    printf("Starting GPU optimization...\n");

    // Validate input pointers
    if (!opt || !opt->context || !opt->queue || !opt->device || 
        !opt->population_buffer || !opt->fitness_buffer || 
        !opt->bounds || !opt->best_solution.position) {
        fprintf(stderr, "Error: Invalid Optimizer structure or null pointers\n");
        exit(1);
    }

    // Read kernel source
    FILE *fp = fopen("FlyFO.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to open FlyFO.cl\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    if (source_size == 0) {
        fprintf(stderr, "Error: FlyFO.cl is empty\n");
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
        fprintf(stderr, "Error: Failed to read FlyFO.cl completely (%zu of %zu bytes)\n", read_size, source_size);
        free(source_str);
        exit(1);
    }
    source_str[source_size] = '\0';

    // Validate kernel source content
    int valid_content = 0;
    for (size_t i = 0; i < source_size; i++) {
        if (!isspace(source_str[i]) && isprint(source_str[i])) {
            valid_content = 1;
            break;
        }
    }
    if (!valid_content) {
        fprintf(stderr, "Error: FlyFO.cl contains only whitespace or invalid characters\n");
        free(source_str);
        exit(1);
    }

    FlyFOCLContext cl_ctx = {0};
    FlyFO_init_cl(&cl_ctx, source_str, opt);
    free(source_str);

    FlyFOContext ctx = {0};
    FlyFO_init_context(&ctx, opt, &cl_ctx);

    int pop_size = opt->population_size;
    int dim = opt->dim;
    int surv_list_size = (int)(pop_size * SURVIVAL_LIST_RATIO);
    int func_count = 0;
    int iteration = 0;

    // Initialize population
    cl_int err;
    cl_mem best_fitness_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer best_fitness");
    cl_mem alpha_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer alpha");
    cl_mem pa_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, pop_size * sizeof(cl_float), NULL, &err);
    check_cl_error(err, "clCreateBuffer pa");
    cl_mem surv_count_buf = clCreateBuffer(cl_ctx.context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
    check_cl_error(err, "clCreateBuffer surv_count");

    clSetKernelArg(cl_ctx.init_pop_kernel, 0, sizeof(cl_mem), &ctx.population);
    clSetKernelArg(cl_ctx.init_pop_kernel, 1, sizeof(cl_mem), &ctx.fitness);
    clSetKernelArg(cl_ctx.init_pop_kernel, 2, sizeof(cl_mem), &ctx.past_fitness);
    clSetKernelArg(cl_ctx.init_pop_kernel, 3, sizeof(cl_mem), &ctx.bounds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 4, sizeof(cl_mem), &ctx.best_position);
    clSetKernelArg(cl_ctx.init_pop_kernel, 5, sizeof(cl_mem), &best_fitness_buf);
    clSetKernelArg(cl_ctx.init_pop_kernel, 6, sizeof(cl_mem), &ctx.random_seeds);
    clSetKernelArg(cl_ctx.init_pop_kernel, 7, sizeof(cl_int), &dim);
    clSetKernelArg(cl_ctx.init_pop_kernel, 8, sizeof(cl_int), &pop_size);

    size_t global_work_size = pop_size;
    err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.init_pop_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueNDRangeKernel init_pop");
    clFinish(cl_ctx.queue);

    // Evaluate initial population on CPU using objective_function
    float *positions = (float *)malloc(pop_size * dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueReadBuffer population");
    float *fitness = (float *)malloc(pop_size * sizeof(float));
    for (int i = 0; i < pop_size; i++) {
        double *pos = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
        fitness[i] = (float)objective_function(pos);
        free(pos);
    }
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer fitness");
    free(positions);

    func_count += pop_size;
    float best_fitness = fitness[0], worst_fitness = fitness[0];
    int best_idx = 0;
    for (int i = 1; i < pop_size; i++) {
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            best_idx = i;
        }
        if (fitness[i] > worst_fitness) worst_fitness = fitness[i];
    }
    ctx.best_fitness = best_fitness;
    ctx.worst_fitness = worst_fitness;
    free(fitness);

    positions = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueReadBuffer best_pos");
    err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer best_pos");
    free(positions);

    err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &best_fitness, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueWriteBuffer best_fitness");

    while (func_count < MAX_EVALS_DEFAULT) {
        iteration++;
        printf("Iteration %d\n", iteration);

        // Fuzzy self-tuning
        clSetKernelArg(cl_ctx.fuzzy_tuning_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.fuzzy_tuning_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.fuzzy_tuning_kernel, 2, sizeof(cl_mem), &ctx.past_fitness);
        clSetKernelArg(cl_ctx.fuzzy_tuning_kernel, 3, sizeof(cl_mem), &ctx.best_position);
        clSetKernelArg(cl_ctx.fuzzy_tuning_kernel, 4, sizeof(cl_float), &ctx.best_fitness);
        clSetKernelArg(cl_ctx.fuzzy_tuning_kernel, 5, sizeof(cl_float), &ctx.worst_fitness);
        clSetKernelArg(cl_ctx.fuzzy_tuning_kernel, 6, sizeof(cl_mem), &alpha_buf);
        clSetKernelArg(cl_ctx.fuzzy_tuning_kernel, 7, sizeof(cl_mem), &pa_buf);
        clSetKernelArg(cl_ctx.fuzzy_tuning_kernel, 8, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.fuzzy_tuning_kernel, 9, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.fuzzy_tuning_kernel, 10, sizeof(cl_int), &pop_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.fuzzy_tuning_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueNDRangeKernel fuzzy_tuning");

        // Update position
        clSetKernelArg(cl_ctx.update_position_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.update_position_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.update_position_kernel, 2, sizeof(cl_mem), &ctx.past_fitness);
        clSetKernelArg(cl_ctx.update_position_kernel, 3, sizeof(cl_mem), &ctx.best_position);
        clSetKernelArg(cl_ctx.update_position_kernel, 4, sizeof(cl_float), &ctx.best_fitness);
        clSetKernelArg(cl_ctx.update_position_kernel, 5, sizeof(cl_float), &ctx.worst_fitness);
        clSetKernelArg(cl_ctx.update_position_kernel, 6, sizeof(cl_mem), &alpha_buf);
        clSetKernelArg(cl_ctx.update_position_kernel, 7, sizeof(cl_mem), &pa_buf);
        clSetKernelArg(cl_ctx.update_position_kernel, 8, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.update_position_kernel, 9, sizeof(cl_mem), &ctx.survival_list);
        clSetKernelArg(cl_ctx.update_position_kernel, 10, sizeof(cl_mem), &ctx.survival_fitness);
        clSetKernelArg(cl_ctx.update_position_kernel, 11, sizeof(cl_mem), &surv_count_buf);
        clSetKernelArg(cl_ctx.update_position_kernel, 12, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.update_position_kernel, 13, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.update_position_kernel, 14, sizeof(cl_int), &pop_size);
        clSetKernelArg(cl_ctx.update_position_kernel, 15, sizeof(cl_int), &surv_list_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.update_position_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueNDRangeKernel update_position");

        // Evaluate population on CPU
        positions = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer population");
        fitness = (float *)malloc(pop_size * sizeof(float));
        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            fitness[i] = (float)objective_function(pos);
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer fitness");
        free(positions);
        func_count += pop_size * 2;

        // Suffocation phase
        int best_count = 0;
        for (int i = 0; i < pop_size; i++) if (fitness[i] == ctx.best_fitness) best_count++;

        clSetKernelArg(cl_ctx.suffocation_kernel, 0, sizeof(cl_mem), &ctx.population);
        clSetKernelArg(cl_ctx.suffocation_kernel, 1, sizeof(cl_mem), &ctx.fitness);
        clSetKernelArg(cl_ctx.suffocation_kernel, 2, sizeof(cl_mem), &ctx.past_fitness);
        clSetKernelArg(cl_ctx.suffocation_kernel, 3, sizeof(cl_mem), &ctx.survival_list);
        clSetKernelArg(cl_ctx.suffocation_kernel, 4, sizeof(cl_mem), &ctx.survival_fitness);
        clSetKernelArg(cl_ctx.suffocation_kernel, 5, sizeof(cl_int), &ctx.survival_count);
        clSetKernelArg(cl_ctx.suffocation_kernel, 6, sizeof(cl_mem), &ctx.bounds);
        clSetKernelArg(cl_ctx.suffocation_kernel, 7, sizeof(cl_float), &ctx.best_fitness);
        clSetKernelArg(cl_ctx.suffocation_kernel, 8, sizeof(cl_mem), &ctx.random_seeds);
        clSetKernelArg(cl_ctx.suffocation_kernel, 9, sizeof(cl_int), &dim);
        clSetKernelArg(cl_ctx.suffocation_kernel, 10, sizeof(cl_int), &pop_size);
        clSetKernelArg(cl_ctx.suffocation_kernel, 11, sizeof(cl_int), &surv_list_size);

        err = clEnqueueNDRangeKernel(cl_ctx.queue, cl_ctx.suffocation_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueNDRangeKernel suffocation");
        func_count += best_count;

        // Update best and worst fitness
        positions = (float *)malloc(pop_size * dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, 0, pop_size * dim * sizeof(cl_float), positions, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer population");
        fitness = (float *)malloc(pop_size * sizeof(float));
        for (int i = 0; i < pop_size; i++) {
            double *pos = (double *)malloc(dim * sizeof(double));
            for (int j = 0; j < dim; j++) pos[j] = positions[i * dim + j];
            fitness[i] = (float)objective_function(pos);
            free(pos);
        }
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.fitness, CL_TRUE, 0, pop_size * sizeof(cl_float), fitness, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer fitness");
        free(positions);

        float new_best = fitness[0], new_worst = fitness[0];
        best_idx = 0;
        for (int i = 1; i < pop_size; i++) {
            if (fitness[i] < new_best) {
                new_best = fitness[i];
                best_idx = i;
            }
            if (fitness[i] > new_worst) new_worst = fitness[i];
        }
        ctx.best_fitness = new_best;
        ctx.worst_fitness = new_worst;

        positions = (float *)malloc(dim * sizeof(float));
        err = clEnqueueReadBuffer(cl_ctx.queue, ctx.population, CL_TRUE, best_idx * dim * sizeof(cl_float), dim * sizeof(cl_float), positions, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueReadBuffer best_pos");
        err = clEnqueueWriteBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), positions, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer best_pos");
        free(positions);
        free(fitness);

        err = clEnqueueWriteBuffer(cl_ctx.queue, best_fitness_buf, CL_TRUE, 0, sizeof(cl_float), &new_best, 0, NULL, NULL);
        check_cl_error(err, "clEnqueueWriteBuffer best_fitness");

        printf("Iteration %d: Function Evaluations = %d, Best Fitness = %f\n", iteration, func_count, ctx.best_fitness);
    }

    // Copy best solution to optimizer
    opt->best_solution.fitness = ctx.best_fitness;
    float *best_pos = (float *)malloc(dim * sizeof(float));
    err = clEnqueueReadBuffer(cl_ctx.queue, ctx.best_position, CL_TRUE, 0, dim * sizeof(cl_float), best_pos, 0, NULL, NULL);
    check_cl_error(err, "clEnqueueReadBuffer final best_pos");
    for (int i = 0; i < dim; i++) opt->best_solution.position[i] = best_pos[i];
    free(best_pos);

    clReleaseMemObject(best_fitness_buf);
    clReleaseMemObject(alpha_buf);
    clReleaseMemObject(pa_buf);
    clReleaseMemObject(surv_count_buf);
    FlyFO_cleanup_context(&ctx, &cl_ctx);
    FlyFO_cleanup_cl(&cl_ctx);

    printf("Optimization completed\n");
}
