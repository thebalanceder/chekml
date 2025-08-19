% Define the Sphere function
sphere_func = @(x) sum(x.^2);

% Define problem dimensionality and search space bounds
dim = 10;
lower_bounds = -5 * ones(1, dim);
upper_bounds = 5 * ones(1, dim);

% Define algorithm parameters
max_iter = 100;
pop_size = 50;
mutation_rate = 0.1;
crossover_rate = 0.8;

% Run the memetic algorithm
[best_solution, best_fitness] = memetic_algorithm(sphere_func, dim, lower_bounds, upper_bounds, max_iter, pop_size, mutation_rate, crossover_rate);

% Display results
disp(['Best solution found: ', num2str(best_solution)]);
disp(['Best fitness found: ', num2str(best_fitness)]);
