#include "KHO.h"
#include "generaloptimizer.h"

double rand_double(double min, double max);

void initialize_krill_positions(KHO_Optimizer *opt)
{
    int i, j;
    Optimizer *base = opt->base;
    for (i = 0; i < base->population_size; i++)
    {
        for (j = 0; j < base->dim; j++)
        {
            double lb = base->bounds[2 * j];
            double ub = base->bounds[2 * j + 1];
            base->population[i].position[j] = lb + rand_double(0.0, 1.0) * (ub - lb);
        }
        base->population[i].fitness = INFINITY;
    }
}

void evaluate_krill_positions(KHO_Optimizer *opt, double (*objective_function)(double *))
{
    int i;
    Optimizer *base = opt->base;
    for (i = 0; i < base->population_size; i++)
    {
        base->population[i].fitness = objective_function(base->population[i].position);
    }
}

void movement_induced_phase(KHO_Optimizer *opt, double w, double Kw_Kgb)
{
    Optimizer *base = opt->base;
    double *Rgb = (double *)malloc(base->dim * sizeof(double));
    double *RR = (double *)malloc(base->dim * base->population_size * sizeof(double));
    double *R = (double *)malloc(base->population_size * sizeof(double));
    int i, j, n, nn;
    double ds, norm_Rgb, alpha_b, alpha_n;

    for (i = 0; i < base->population_size; i++)
    {
        for (j = 0; j < base->dim; j++)
        {
            Rgb[j] = base->best_solution.position[j] - base->population[i].position[j];
            for (n = 0; n < base->population_size; n++)
            {
                RR[j * base->population_size + n] = base->population[n].position[j] - base->population[i].position[j];
            }
        }

        for (n = 0; n < base->population_size; n++)
        {
            R[n] = 0.0;
            for (j = 0; j < base->dim; j++)
            {
                double diff = RR[j * base->population_size + n];
                R[n] += diff * diff;
            }
            R[n] = sqrt(R[n]);
        }

        alpha_b = 0.0;
        if (base->best_solution.fitness < base->population[i].fitness)
        {
            norm_Rgb = 0.0;
            for (j = 0; j < base->dim; j++)
            {
                norm_Rgb += Rgb[j] * Rgb[j];
            }
            norm_Rgb = sqrt(norm_Rgb);
            if (norm_Rgb > 1e-10)
            {
                alpha_b = -2.0 * (1.0 + rand_double(0.0, 1.0) * ((double)opt->current_iter / base->max_iter)) *
                          (base->best_solution.fitness - base->population[i].fitness) / Kw_Kgb / norm_Rgb;
            }
        }

        alpha_n = 0.0;
        nn = 0;
        ds = 0.0;
        for (n = 0; n < base->population_size; n++)
        {
            ds += R[n];
        }
        ds /= (base->population_size * SENSE_DISTANCE_FACTOR);

        for (n = 0; n < base->population_size && nn < NEIGHBOR_LIMIT; n++)
        {
            if (R[n] < ds && n != i)
            {
                if (base->population[i].fitness != base->population[n].fitness)
                {
                    alpha_n -= (base->population[n].fitness - base->population[i].fitness) / Kw_Kgb / (R[n] + 1e-10);
                }
                nn++;
            }
        }

        for (j = 0; j < base->dim; j++)
        {
            double motion = alpha_b * Rgb[j];
            nn = 0;
            for (n = 0; n < base->population_size && nn < NEIGHBOR_LIMIT; n++)
            {
                if (R[n] < ds && n != i)
                {
                    motion += alpha_n * RR[j * base->population_size + n];
                    nn++;
                }
            }
            opt->solutions[i].N[j] = w * opt->solutions[i].N[j] + NMAX * motion;
        }
    }

    free(Rgb);
    free(RR);
    free(R);
}

void foraging_motion_phase(KHO_Optimizer *opt, double w, double Kf, double *Xf, double Kw_Kgb)
{
    Optimizer *base = opt->base;
    double *Rf = (double *)malloc(base->dim * sizeof(double));
    double *Rib = (double *)malloc(base->dim * sizeof(double));
    int i, j;
    double norm_Rf, norm_Rib, Beta_f, Beta_b;

    for (i = 0; i < base->population_size; i++)
    {
        for (j = 0; j < base->dim; j++)
        {
            Rf[j] = Xf[j] - base->population[i].position[j];
            Rib[j] = opt->solutions[i].best_position[j] - base->population[i].position[j];
        }

        Beta_f = 0.0;
        if (Kf < base->population[i].fitness)
        {
            norm_Rf = 0.0;
            for (j = 0; j < base->dim; j++)
            {
                norm_Rf += Rf[j] * Rf[j];
            }
            norm_Rf = sqrt(norm_Rf);
            if (norm_Rf > 1e-10)
            {
                Beta_f = -2.0 * (1.0 - (double)opt->current_iter / base->max_iter) *
                         (Kf - base->population[i].fitness) / Kw_Kgb / norm_Rf;
            }
        }

        Beta_b = 0.0;
        if (opt->solutions[i].best_fitness < base->population[i].fitness)
        {
            norm_Rib = 0.0;
            for (j = 0; j < base->dim; j++)
            {
                norm_Rib += Rib[j] * Rib[j];
            }
            norm_Rib = sqrt(norm_Rib);
            if (norm_Rib > 1e-10)
            {
                Beta_b = -(opt->solutions[i].best_fitness - base->population[i].fitness) / Kw_Kgb / norm_Rib;
            }
        }

        for (j = 0; j < base->dim; j++)
        {
            opt->solutions[i].F[j] = w * opt->solutions[i].F[j] + VF * (Beta_f * Rf[j] + Beta_b * Rib[j]);
        }
    }

    free(Rf);
    free(Rib);
}

void physical_diffusion_phase(KHO_Optimizer *opt, int iteration, double Kw_Kgb)
{
    Optimizer *base = opt->base;
    int i, j;
    double diffusion;

    for (i = 0; i < base->population_size; i++)
    {
        diffusion = DMAX * (1.0 - (double)iteration / base->max_iter) *
                    (rand_double(0.0, 1.0) + (base->population[i].fitness - base->best_solution.fitness) / Kw_Kgb);
        for (j = 0; j < base->dim; j++)
        {
            opt->solutions[i].D[j] = diffusion * (2.0 * rand_double(0.0, 1.0) - 1.0);
        }
    }
}

void crossover_phase(KHO_Optimizer *opt, double Kw_Kgb)
{
    Optimizer *base = opt->base;
    int i, j, NK4Cr;
    double C_rate;

    for (i = 0; i < base->population_size; i++)
    {
        C_rate = CROSSOVER_RATE + CROSSOVER_SCALE * (base->population[i].fitness - base->best_solution.fitness) / Kw_Kgb;
        NK4Cr = (int)(rand_double(0.0, 1.0) * (base->population_size - 1));
        for (j = 0; j < base->dim; j++)
        {
            if (rand_double(0.0, 1.0) < C_rate)
            {
                base->population[i].position[j] = base->population[NK4Cr].position[j];
            }
        }
    }
}

void kho_update_positions(KHO_Optimizer *opt)
{
    Optimizer *base = opt->base;
    int i, j;

    for (i = 0; i < base->population_size; i++)
    {
        for (j = 0; j < base->dim; j++)
        {
            base->population[i].position[j] += opt->Dt * (opt->solutions[i].N[j] + opt->solutions[i].F[j] + opt->solutions[i].D[j]);
        }
    }
}

void enforce_kho_bounds(KHO_Optimizer *opt, double *position, double *best)
{
    Optimizer *base = opt->base;
    int j;
    double lb, ub, A, B;

    for (j = 0; j < base->dim; j++)
    {
        lb = base->bounds[2 * j];
        ub = base->bounds[2 * j + 1];
        if (position[j] < lb)
        {
            A = rand_double(0.0, 1.0);
            position[j] = A * lb + (1.0 - A) * best[j];
        }
        else if (position[j] > ub)
        {
            B = rand_double(0.0, 1.0);
            position[j] = B * ub + (1.0 - B) * best[j];
        }
    }
}

void KHO_optimize(Optimizer *base, double (*objective_function)(double *))
{
    int i, j, best_idx;
    KHO_Optimizer opt;
    opt.base = base;
    opt.current_iter = 0;
    opt.crossover_flag = 1;
    double *Xf = (double *)malloc(base->dim * sizeof(double));
    double *Sf = (double *)malloc(base->dim * sizeof(double));
    double *K = (double *)malloc(base->population_size * sizeof(double));
    double *Kib = (double *)malloc(base->population_size * sizeof(double));
    double *Xib = (double *)malloc(base->dim * base->population_size * sizeof(double));
    double bounds_diff_sum = 0.0;
    double sum_inv_K, w, max_K, Kw_Kgb, Kf;

    for (j = 0; j < base->dim; j++)
    {
        bounds_diff_sum += fabs(base->bounds[2 * j + 1] - base->bounds[2 * j]);
    }
    opt.Dt = bounds_diff_sum / (2.0 * base->dim);

    opt.solutions = (KHO_Solution *)malloc(base->population_size * sizeof(KHO_Solution));
    for (i = 0; i < base->population_size; i++)
    {
        opt.solutions[i].best_fitness = INFINITY;
        opt.solutions[i].best_position = (double *)malloc(base->dim * sizeof(double));
        opt.solutions[i].N = (double *)malloc(base->dim * sizeof(double));
        opt.solutions[i].F = (double *)malloc(base->dim * sizeof(double));
        opt.solutions[i].D = (double *)malloc(base->dim * sizeof(double));
        for (j = 0; j < base->dim; j++)
        {
            opt.solutions[i].best_position[j] = base->population[i].position[j];
            opt.solutions[i].N[j] = 0.0;
            opt.solutions[i].F[j] = 0.0;
            opt.solutions[i].D[j] = 0.0;
        }
    }

    opt.history = (KHO_History *)malloc(base->max_iter * sizeof(KHO_History));
    for (i = 0; i < base->max_iter; i++)
    {
        opt.history[i].position = (double *)malloc(base->dim * sizeof(double));
        opt.history[i].fitness = INFINITY;
    }

    initialize_krill_positions(&opt);
    evaluate_krill_positions(&opt, objective_function);

    for (i = 0; i < base->population_size; i++)
    {
        K[i] = base->population[i].fitness;
        Kib[i] = K[i];
        for (j = 0; j < base->dim; j++)
        {
            Xib[j * base->population_size + i] = base->population[i].position[j];
            opt.solutions[i].best_position[j] = base->population[i].position[j];
        }
        opt.solutions[i].best_fitness = K[i];
    }

    best_idx = 0;
    for (i = 1; i < base->population_size; i++)
    {
        if (K[i] < K[best_idx])
        {
            best_idx = i;
        }
    }
    base->best_solution.fitness = K[best_idx];
    for (j = 0; j < base->dim; j++)
    {
        base->best_solution.position[j] = base->population[best_idx].position[j];
    }

    for (opt.current_iter = 0; opt.current_iter < base->max_iter; opt.current_iter++)
    {
        sum_inv_K = 0.0;
        for (i = 0; i < base->population_size; i++)
        {
            sum_inv_K += 1.0 / (K[i] + 1e-10);
        }
        for (j = 0; j < base->dim; j++)
        {
            Sf[j] = 0.0;
            for (i = 0; i < base->population_size; i++)
            {
                Sf[j] += base->population[i].position[j] / (K[i] + 1e-10);
            }
            Xf[j] = Sf[j] / sum_inv_K;
        }

        enforce_kho_bounds(&opt, Xf, base->best_solution.position);
        Kf = objective_function(Xf);

        if (opt.current_iter > 0 && Kf < opt.history[opt.current_iter - 1].fitness)
        {
            for (j = 0; j < base->dim; j++)
            {
                Xf[j] = opt.history[opt.current_iter - 1].position[j];
            }
            Kf = opt.history[opt.current_iter - 1].fitness;
        }

        w = INERTIA_MIN + INERTIA_MAX * (1.0 - (double)opt.current_iter / base->max_iter);
        max_K = K[0];
        for (i = 1; i < base->population_size; i++)
        {
            if (K[i] > max_K)
            {
                max_K = K[i];
            }
        }
        Kw_Kgb = max_K - base->best_solution.fitness;

        movement_induced_phase(&opt, w, Kw_Kgb);
        foraging_motion_phase(&opt, w, Kf, Xf, Kw_Kgb);
        physical_diffusion_phase(&opt, opt.current_iter, Kw_Kgb);
        if (opt.crossover_flag)
        {
            crossover_phase(&opt, Kw_Kgb);
        }
        kho_update_positions(&opt);

        for (i = 0; i < base->population_size; i++)
        {
            enforce_kho_bounds(&opt, base->population[i].position, base->best_solution.position);
            K[i] = objective_function(base->population[i].position);
            if (K[i] < Kib[i])
            {
                Kib[i] = K[i];
                for (j = 0; j < base->dim; j++)
                {
                    Xib[j * base->population_size + i] = base->population[i].position[j];
                    opt.solutions[i].best_position[j] = base->population[i].position[j];
                }
                opt.solutions[i].best_fitness = K[i];
            }
        }

        best_idx = 0;
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
            for (j = 0; j < base->dim; j++)
            {
                base->best_solution.position[j] = base->population[best_idx].position[j];
            }
        }

        opt.history[opt.current_iter].fitness = base->best_solution.fitness;
        for (j = 0; j < base->dim; j++)
        {
            opt.history[opt.current_iter].position[j] = base->best_solution.position[j];
        }
        printf("Iteration %d: Best Value = %f\n", opt.current_iter + 1, base->best_solution.fitness);
    }

    for (i = 0; i < base->population_size; i++)
    {
        free(opt.solutions[i].best_position);
        free(opt.solutions[i].N);
        free(opt.solutions[i].F);
        free(opt.solutions[i].D);
    }
    for (i = 0; i < base->max_iter; i++)
    {
        free(opt.history[i].position);
    }
    free(opt.solutions);
    free(opt.history);
    free(Xf);
    free(Sf);
    free(K);
    free(Kib);
    free(Xib);
}
