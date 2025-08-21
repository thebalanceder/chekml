__kernel void evaluate_fitness(
    __global const float* population,
    __global float* fitness,
    const int dim,
    const int population_size)
{
    int id = get_global_id(0);
    if (id < population_size) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            float x = population[id * dim + d];
            sum += x * x; // Sphere function as fallback
        }
        fitness[id] = sum;
    }
}
