% Central Force Optimization (CFO) Implementation


% Parameters
population_size = 50;    % Population size
num_variables = 10;      % Number of variables
max_iterations = 100;    % Maximum number of iterations
alpha = 0.1;              % Learning rate
lower_limit = -5;        % Lower limit for variables
upper_limit = 5;         % Upper limit for variables

% Initialization
population = lower_limit + (upper_limit - lower_limit) * rand(population_size, num_variables);  % Random initial population

% Main loop
for iter = 1:max_iterations
    % Evaluate objective function for each individual
    fitness = zeros(population_size, 1);
    for i = 1:population_size
        fitness(i) = objective_function(population(i, :));
    end
    
    % Sort population based on fitness
    [fitness, sorted_indices] = sort(fitness);
    population = population(sorted_indices, :);
    
    % Calculate center of mass
    center_of_mass = mean(population);
    
    % Update each individual's position
    for i = 1:population_size
        direction = center_of_mass - population(i, :);
        population(i, :) = population(i, :) + alpha * direction;
        
        % Ensure the new position is within bounds
        population(i, :) = max(min(population(i, :), upper_limit), lower_limit);
    end
end

% Find the best solution
[best_fitness, best_index] = min(fitness);
best_solution = population(best_index, :);
fprintf('Best solution: %s\n', mat2str(best_solution));
fprintf('Best fitness: %f\n', best_fitness);

% Objective function (example)
function z = objective_function(x)
    % Example: Sphere function
    z = sum(x.^2);
end
