% Locust Swarm Algorithm (LSA) Implementation
% Parameters
population_size = 50;        % Population size
num_variables = 10;          % Number of variables
max_iterations = 100;        % Maximum number of iterations
step_size = 0.1;              % Step size for movement
lower_limit = -5;            % Lower limit for variables
upper_limit = 5;             % Upper limit for variables

% Initialization
population = lower_limit + (upper_limit - lower_limit) * rand(population_size, num_variables);  % Random initial population
fitness = zeros(population_size, 1);  % Initialize fitness values

% Main loop
for iter = 1:max_iterations
    % Evaluate fitness for each individual in the population
    for i = 1:population_size
        fitness(i) = your_objective_function(population(i, :)); % Evaluate using your objective function
    end
    
    % Find the best individual in the population
    [best_fitness, best_index] = min(fitness);
    best_solution = population(best_index, :);
    
    % Update each individual's position
    for i = 1:population_size
        % Move each individual towards the best solution
        direction = best_solution - population(i, :);
        population(i, :) = population(i, :) + step_size * direction * rand();
        
        % Ensure the new position is within bounds
        population(i, :) = max(min(population(i, :), upper_limit), lower_limit);
    end
end

% Display best solution found
fprintf('Best solution: %s\n', mat2str(best_solution));
fprintf('Best fitness: %f\n', best_fitness);
% Define your objective function here
function z = your_objective_function(x)
    % Example: Sphere function
    z = sum(x.^2);
end