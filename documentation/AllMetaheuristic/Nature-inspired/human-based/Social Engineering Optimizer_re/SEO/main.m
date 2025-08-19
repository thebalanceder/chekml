% Define the objective function (Sphere function)
sphere_func = @(x) sum(x.^2);

% Parameters
num_variables = 10; % Number of variables
num_agents = 50; % Number of agents
max_iter = 100; % Maximum number of iterations
lb = -5 * ones(1, num_variables); % Lower bounds
ub = 5 * ones(1, num_variables); % Upper bounds

% Run Social Engineering Optimizer
[best_solution, best_fitness] = seo(sphere_func, num_variables, num_agents, max_iter, lb, ub);

% Display results
disp('Best solution:');
disp(best_solution);
disp('Best fitness:');
disp(best_fitness);
