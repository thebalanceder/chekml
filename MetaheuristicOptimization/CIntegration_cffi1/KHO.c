#include "KHO.h"
#include "generaloptimizer.h"

double rand_double(double min, double max);

void initialize_krill_positions(KHO_Optimizer *opt)
{
    Optimizer *base = opt->base;
    double *bounds = base->bounds;
    int i, j;
    if (base->dim > KHO_MAX_DIM || base->population_size > KHO_MAX_POP)
    {
        fprintf(stderr, "Error: dim or population_size exceeds KHO limits\n");
        exit(1);
    }
    #pragma omp simd
    for (i = 0; i < base->population_size; i++)
    {
        double *pos = base->population[i].position;
        for (j = 0; j < base->dim; j++)
        {
            double lb = bounds[j << 1];
            double ub = bounds[(j << 1) + 1];
            pos[j] = lb + rand_double(0.0, 1.0) * (ub - lb);
        }
        base->population[i].fitness = INFINITY;
    }
}

void evaluate_krill_positions(KHO_Optimizer *opt, double (*objective_function)(double *))
{
    Optimizer *base = opt->base;
    double *K = opt->buffers.K;
    int i;
    #pragma omp simd
    for (i = 0; i < base->population_size; i++)
    {
        K[i] = objective_function(base->population[i].position);
        base->population[i].fitness = K[i];
    }
}

void movement_induced_phase(KHO_Optimizer *opt, double w, double Kw_Kgb)
{
    Optimizer *base = opt->base;
    KHO_Buffers *buf = &opt->buffers;
    double *Rgb = buf->Rgb;
    double *RR = buf->RR;
    double *R = buf->R;
    double inv_Kw_Kgb = Kw_Kgb > 1e-10 ? 1.0 / Kw_Kgb : 0.0;
    double t = (double)opt->current_iter * opt->inv_max_iter;
    int i, j, n;

    for (i = 0; i < base->population_size; i++)
    {
        double *pos_i = base->population[i].position;
        double fit_i = base->population[i].fitness;
        double norm_Rgb = 0.0;
        double ds = 0.0;

        for (j = 0; j < base->dim; j++)
        {
            Rgb[j] = base->best_solution.position[j] - pos_i[j];
            norm_Rgb += Rgb[j] * Rgb[j];
            double diff;
            #pragma omp simd
            for (n = 0; n < base->population_size; n++)
            {
                diff = base->population[n].position[j] - pos_i[j];
                RR[j * base->population_size + n] = diff;
            }
        }
        norm_Rgb = sqrt(norm_Rgb);

        #pragma omp simd
        for (n = 0; n < base->population_size; n++)
        {
            double sum = 0.0;
            for (j = 0; j < base->dim; j++)
            {
                double diff = RR[j * base->population_size + n];
                sum += diff * diff;
            }
            R[n] = sqrt(sum);
            ds += R[n];
        }
        ds *= 1.0 / (base->population_size * SENSE_DISTANCE_FACTOR);

        double alpha_b = 0.0;
        if (base->best_solution.fitness < fit_i && norm_Rgb > 1e-10)
        {
            alpha_b = -2.0 * (1.0 + rand_double(0.0, 1.0) * t) *
                      (base->best_solution.fitness - fit_i) * inv_Kw_Kgb / norm_Rgb;
        }

        double alpha_n = 0.0;
        int nn = 0;
        for (n = 0; n < base->population_size && nn < NEIGHBOR_LIMIT; n++)
        {
            if (R[n] < ds && n != i)
            {
                double fit_n = base->population[n].fitness;
                alpha_n -= (fit_n - fit_i) * inv_Kw_Kgb / (R[n] + 1e-10);
                nn++;
            }
        }

        double *N = opt->solutions[i].N;
        #pragma omp simd
        for (j = 0; j < base->dim; j++)
        {
            double motion = alpha_b * Rgb[j];
            if (nn > 0)
            {
                motion += alpha_n * RR[j * base->population_size + i];
            }
            N[j] = w * N[j] + NMAX * motion;
        }
    }
}

void foraging_motion_phase(KHO_Optimizer *opt, double w, double Kf, double Kw_Kgb)
{
    Optimizer *base = opt->base;
    KHO_Buffers *buf = &opt->buffers;
    double *Rf = buf->Rf;
    double *Rib = buf->Rib;
    double *Xf = buf->Xf;
    double inv_Kw_Kgb = Kw_Kgb > 1e-10 ? 1.0 / Kw_Kgb : 0.0;
    double t = 1.0 - (double)opt->current_iter * opt->inv_max_iter;
    int i, j;

    for (i = 0; i < base->population_size; i++)
    {
        double *pos_i = base->population[i].position;
        double fit_i = base->population[i].fitness;
        KHO_Solution *sol = &opt->solutions[i];
        double norm_Rf = 0.0;
        double norm_Rib = 0.0;

        #pragma omp simd
        for (j = 0; j < base->dim; j++)
        {
            Rf[j] = Xf[j] - pos_i[j];
            Rib[j] = sol->best_position[j] - pos_i[j];
            norm_Rf += Rf[j] * Rf[j];
            norm_Rib += Rib[j] * Rib[j];
        }
        norm_Rf = sqrt(norm_Rf);
        norm_Rib = sqrt(norm_Rib);

        double Beta_f = norm_Rf > 1e-10 ? -2.0 * t * (Kf - fit_i) * inv_Kw_Kgb / norm_Rf : 0.0;
        double Beta_b = norm_Rib > 1e-10 ? -(sol->best_fitness - fit_i) * inv_Kw_Kgb / norm_Rib : 0.0;

        double *F = sol->F;
        #pragma omp simd
        for (j = 0; j < base->dim; j++)
        {
            F[j] = w * F[j] + VF * (Beta_f * Rf[j] + Beta_b * Rib[j]);
        }
    }
}

void physical_diffusion_phase(KHO_Optimizer *opt, int iteration, double Kw_Kgb)
{
    Optimizer *base = opt->base;
    double scale = DMAX * (1.0 - (double)iteration * opt->inv_max_iter);
    double inv_Kw_Kgb = Kw_Kgb > 1e-10 ? 1.0 / Kw_Kgb : 0.0;
    int i, j;

    for (i = 0; i < base->population_size; i++)
    {
        double diff = (base->population[i].fitness - base->best_solution.fitness) * inv_Kw_Kgb;
        double diffusion = scale * (rand_double(0.0, 1.0) + diff);
        double *D = opt->solutions[i].D;
        #pragma omp simd
        for (j = 0; j < base->dim; j++)
        {
            D[j] = diffusion * (2.0 * rand_double(0.0, 1.0) - 1.0);
        }
    }
}

void crossover_phase(KHO_Optimizer *opt, double Kw_Kgb)
{
    Optimizer *base = opt->base;
    double inv_Kw_Kgb = Kw_Kgb > 1e-10 ? 1.0 / Kw_Kgb : 0.0;
    int i, j;

    #pragma omp simd
    for (i = 0; i < base->population_size; i++)
    {
        double C_rate = CROSSOVER_RATE + CROSSOVER_SCALE * (base->population[i].fitness - base->best_solution.fitness) * inv_Kw_Kgb;
        int NK4Cr = (int)(rand_double(0.0, 1.0) * (base->population_size - 1));
        double *pos_i = base->population[NK4Cr].position;
        double *pos = base->population[i].position;
        for (j = 0; j < base->dim; j++)
        {
            pos[j] = rand_double(0.0, 1.0) < C_rate ? pos_i[j] : pos[j];
        }
    }
}

void kho_update_positions(KHO_Optimizer *opt)
{
    Optimizer *base = opt->base;
    double Dt = opt->Dt;
    int i, j;

    #pragma omp simd
    for (i = 0; i < base->population_size; i++)
    {
        double *pos = base->population[i].position;
        double *N = opt->solutions[i].N;
        double *F = opt->solutions[i].F;
        double *D = opt->solutions[i].D;
        for (j = 0; j < base->dim; j++)
        {
            pos[j] += Dt * (N[j] + F[j] + D[j]);
        }
    }
}

void enforce_kho_bounds(KHO_Optimizer *opt, double *position, double *best)
{
    Optimizer *base = opt->base;
    double *bounds = base->bounds;
    int j;

    #pragma omp simd
    for (j = 0; j < base->dim; j++)
    {
        double lb = bounds[j << 1];
        double ub = bounds[(j << 1) + 1];
        double p = position[j];
        double r = rand_double(0.0, 1.0);
        position[j] = p < lb ? r * lb + (1.0 - r) * best[j] :
                      p > ub ? r * ub + (1.0 - r) * best[j] : p;
    }
}

void KHO_optimize(Optimizer *base, double (*objective_function)(double *))
{
    KHO_Optimizer *opt = calloc(1, sizeof(KHO_Optimizer));
    if (!opt)
    {
        fprintf(stderr, "Error: Failed to allocate KHO_Optimizer\n");
        exit(1);
    }
    opt->base = base;
    opt->current_iter = 0;
    opt->crossover_flag = 1;
    if (base->max_iter > KHO_MAX_ITER)
    {
        fprintf(stderr, "Error: max_iter exceeds KHO_MAX_ITER\n");
        free(opt);
        exit(1);
    }
    opt->inv_max_iter = 1.0 / base->max_iter;

    double *bounds = base->bounds;
    double Dt = 0.0;
    int j;
    for (j = 0; j < base->dim; j++)
    {
        Dt += fabs(bounds[(j << 1) + 1] - bounds[j << 1]);
    }
    opt->Dt = Dt / (2.0 * base->dim);

    initialize_krill_positions(opt);
    evaluate_krill_positions(opt, objective_function);

    double *K = opt->buffers.K;
    double *Kib = opt->buffers.Kib;
    int i;
    for (i = 0; i < base->population_size; i++)
    {
        Kib[i] = K[i];
        opt->solutions[i].best_fitness = K[i];
        double *best_pos = opt->solutions[i].best_position;
        double *pos = base->population[i].position;
        #pragma omp simd
        for (j = 0; j < base->dim; j++)
        {
            best_pos[j] = pos[j];
        }
    }

    int best_idx = 0;
    #pragma omp simd
    for (i = 1; i < base->population_size; i++)
    {
        if (K[i] < K[best_idx])
        {
            best_idx = i;
        }
    }
    base->best_solution.fitness = K[best_idx];
    double *best_pos = base->best_solution.position;
    double *pop_pos = base->population[best_idx].position;
    #pragma omp simd
    for (j = 0; j < base->dim; j++)
    {
        best_pos[j] = pop_pos[j];
    }

    double *Xf = opt->buffers.Xf;
    double *Sf = opt->buffers.Sf;
    for (opt->current_iter = 0; opt->current_iter < base->max_iter; opt->current_iter++)
    {
        double sum_inv_K = 0.0;
        #pragma omp simd
        for (i = 0; i < base->population_size; i++)
        {
            sum_inv_K += 1.0 / (K[i] + 1e-10);
        }
        double inv_sum = 1.0 / sum_inv_K;
        #pragma omp simd
        for (j = 0; j < base->dim; j++)
        {
            double sum = 0.0;
            for (i = 0; i < base->population_size; i++)
            {
                sum += base->population[i].position[j] / (K[i] + 1e-10);
            }
            Xf[j] = sum * inv_sum;
        }

        enforce_kho_bounds(opt, Xf, base->best_solution.position);
        double Kf = objective_function(Xf);

        if (opt->current_iter > 0 && Kf < opt->history[opt->current_iter - 1].fitness)
        {
            #pragma omp simd
            for (j = 0; j < base->dim; j++)
            {
                Xf[j] = opt->history[opt->current_iter - 1].position[j];
            }
            Kf = opt->history[opt->current_iter - 1].fitness;
        }

        double w = INERTIA_MIN + INERTIA_MAX * (1.0 - opt->current_iter * opt->inv_max_iter);
        double max_K = K[0];
        #pragma omp simd
        for (i = 1; i < base->population_size; i++)
        {
            if (K[i] > max_K)
            {
                max_K = K[i];
            }
        }
        double Kw_Kgb = max_K - base->best_solution.fitness;

        movement_induced_phase(opt, w, Kw_Kgb);
        foraging_motion_phase(opt, w, Kf, Kw_Kgb);
        physical_diffusion_phase(opt, opt->current_iter, Kw_Kgb);
        if (opt->crossover_flag)
        {
            crossover_phase(opt, Kw_Kgb);
        }
        kho_update_positions(opt);

        for (i = 0; i < base->population_size; i++)
        {
            double *pos = base->population[i].position;
            enforce_kho_bounds(opt, pos, base->best_solution.position);
            K[i] = objective_function(pos);
            base->population[i].fitness = K[i];
            if (K[i] < Kib[i])
            {
                Kib[i] = K[i];
                opt->solutions[i].best_fitness = K[i];
                double *best_pos_i = opt->solutions[i].best_position;
                #pragma omp simd
                for (j = 0; j < base->dim; j++)
                {
                    best_pos_i[j] = pos[j];
                }
            }
        }

        best_idx = 0;
        #pragma omp simd
        for (i = 1; i < base->population_size; i++)
        {
            if (K[i] < K[best_idx])
            {
                best_idx = i;
            }
        }
        if (K[best_idx] < base->best_solution.fitness)
        {
            base->best_solution.fitness = K[best_idx];
            best_pos = base->best_solution.position;
            pop_pos = base->population[best_idx].position;
            #pragma omp simd
            for (j = 0; j < base->dim; j++)
            {
                best_pos[j] = pop_pos[j];
            }
        }

        KHO_History *hist = &opt->history[opt->current_iter];
        hist->fitness = base->best_solution.fitness;
        #pragma omp simd
        for (j = 0; j < base->dim; j++)
        {
            hist->position[j] = base->best_solution.position[j];
        }
        printf("Iteration %d: Best Value = %f\n", opt->current_iter + 1, base->best_solution.fitness);
    }

    free(opt);
}
