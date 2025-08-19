% Parameters
num_geese = 20;                % Number of geese (solutions)
num_dimensions = 5;            % Number of dimensions (variables)
max_iterations = 100;          % Maximum number of iterations
alpha = 0.9;                    % Scaling factor
beta = 0.1;                     % Learning rate
gamma = 0.1;                    % Randomization parameter
lb = -10;                       % Lower bounds for variables
ub = 10;                        % Upper bounds for variables

% Initialization: Randomly initialize goose positions
geese = rand(num_geese, num_dimensions) * (ub - lb) + lb;
fitness = zeros(num_geese, 1);  % Fitness values

% Main loop (Wild Geese Migration Optimization Algorithm)
for iteration = 1:max_iterations
    % Evaluate fitness for each goose
    for i = 1:num_geese
        fitness(i) = sphere_function(geese(i, :));
    end
    
    % Sort geese based on fitness
    [fitness, sorted_indices] = sort(fitness);
    geese = geese(sorted_indices, :);
    
    % Update goose positions
    for i = 1:num_geese
        % Determine the best goose
        best_goose = geese(1, :);
        % Update position
        geese(i, :) = alpha * geese(i, :) + beta * rand(1, num_dimensions) .* (best_goose - geese(i, :)) + gamma * rand(1, num_dimensions) .* (ub - lb);
        % Ensure positions stay within bounds [lb, ub]
        geese(i, :) = min(max(geese(i, :), lb), ub);
    end
    
    % Display best solution
    best_solution = geese(1, :);
    best_fitness = fitness(1);
    fprintf('Iteration %d: Best Fitness = %f\n', iteration, best_fitness);
end

% Display final result
fprintf('Optimization finished.\n');
fprintf('Best solution found: %s\n', mat2str(best_solution));
fprintf('Best fitness: %f\n', best_fitness);
