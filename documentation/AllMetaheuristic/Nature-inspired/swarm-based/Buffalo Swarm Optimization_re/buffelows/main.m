% Define parameters
num_buffaloes = 50;
num_iterations = 100;
num_variables = 3;
lower_bound = -5;
upper_bound = 5;

% Run Buffalo Swarm Optimization
[best_solution, best_fitness] = buffalo_swarm_optimization(num_buffaloes, num_iterations, num_variables, lower_bound, upper_bound);

% Display results
disp('Optimization complete.');
disp(['Best solution: ', num2str(best_solution)]);
disp(['Best fitness: ', num2str(best_fitness)]);
