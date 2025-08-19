% Coral Reefs Optimization (CRO) Implementation

% Parameters
population_size = 50;        % Population size
num_variables = 10;          % Number of variables
max_iterations = 100;        % Maximum number of iterations
num_reefs = 10;              % Number of reefs (subpopulations)
alpha = 0.1;                  % Learning rate
lower_limit = -5;            % Lower limit for variables
upper_limit = 5;             % Upper limit for variables

% Initialization
population = cell(num_reefs, 1);
for i = 1:num_reefs
    population{i} = lower_limit + (upper_limit - lower_limit) * rand(population_size, num_variables);  % Random initial population
end

% Main loop
for iter = 1:max_iterations
    % Evaluate fitness for each reef
    fitness = zeros(num_reefs, population_size);
    for i = 1:num_reefs
        for j = 1:population_size
            fitness(i, j) = objective_function(population{i}(j, :));
        end
    end
    
    % Exchange solutions between reefs (migration)
    for i = 1:num_reefs
        for j = 1:num_reefs
            if i ~= j
                % Randomly select a solution to migrate
                idx = randi([1, population_size]);
                solution_to_migrate = population{i}(idx, :);
                
                % Replace a random solution in the destination reef
                idx_replace = randi([1, population_size]);
                population{j}(idx_replace, :) = solution_to_migrate;
            end
        end
    end
    
    % Update each reef's population
    for i = 1:num_reefs
        for j = 1:population_size
            % Perform local search (e.g., gradient descent) on each solution
            % Here, we use a simple random perturbation
            population{i}(j, :) = population{i}(j, :) + alpha * randn(1, num_variables);
            
            % Ensure the new position is within bounds
            population{i}(j, :) = max(min(population{i}(j, :), upper_limit), lower_limit);
        end
    end
end

% Find the best solution among all reefs
best_fitness = inf;
best_solution = [];
for i = 1:num_reefs
    for j = 1:population_size
        if objective_function(population{i}(j, :)) < best_fitness
            best_fitness = objective_function(population{i}(j, :));
            best_solution = population{i}(j, :);
        end
    end
end

% Display best solution
fprintf('Best solution: %s\n', mat2str(best_solution));
fprintf('Best fitness: %f\n', best_fitness);
% Objective function (example)
function z = objective_function(x)
    % Example: Sphere function
    z = sum(x.^2);
end