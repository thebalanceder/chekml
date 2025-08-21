#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "generaloptimizer.h"
#include "DISOwithRCF.h"  // Include DISO with RCF header

// ✅ Schwefel Function Definition
double schwefel_function(double *x) {
    int i;
    double sum = 0.0;
    int dim = 2;  // Make sure to match optimizer's dimension

    for (i = 0; i < dim; i++) {
        sum += x[i] * sin(sqrt(fabs(x[i])));
    }
    return 418.9829 * dim - sum;  // Global minimum at ~ -8379.6 for dim=10
}

int main() {
    int dim = 2;                 // Number of dimensions
    int population_size = 50;      // Population size
    int max_iter = 100;           // Number of iterations
    double bounds[] = {-5, 5,-5,5}; // Schwefel domain [-500, 500]

    // ✅ Initialize optimizer (DISO)
    Optimizer *opt = general_init(dim, population_size, max_iter, bounds, "DISO");
    if (!opt) {
        printf("❌ Error initializing optimizer!\n");
        return 1;
    }

    // ✅ Run optimization using DISO
    general_optimize(opt, schwefel_function);

    // ✅ Print best solution
    printf("\n✅ [C] Best Fitness: %f\n", opt->best_solution.fitness);
    printf("✅ [C] Best Solution: ");
    for (int i = 0; i < dim; i++) {
        printf("%f ", opt->best_solution.position[i]);
    }
    printf("\n");

    // ✅ Free memory
    general_free(opt);

    return 0;
}
