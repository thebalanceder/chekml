% Jaguar Algorithm (JA) Implementation

% Parameters
population_size = 50;        % Population size
num_variables = 10;          % Number of variables
max_iterations = 100;        % Maximum number of iterations
p_cruise = 0.8;              % Probability of cruising
cruising_distance = 0.1;     % Maximum cruising distance
alpha = 0.1;                 % Learning rate
lower_limit = -5;            % Lower limit for variables
upper_limit = 5;             % Upper limit for variables

% Initialization
population = lower_limit + (upper_limit - lower_limit) * rand(population_size, num_variables);  % Random initial population

% Main loop
for iter = 1:max_iterations
    % Evaluate fitness for each individual in the population
    fitness = zeros(population_size, 1);
    for i = 1:population_size
        fitness(i) = objective_function(population(i, :));
    end
    
    % Sort population based on fitness
    [fitness, sorted_indices] = sort(fitness);
    population = population(sorted_indices, :);
    
    % Determine cruising individuals
    num_cruising = round(p_cruise * population_size);
    cruising_population = population(1:num_cruising, :);
    
    % Update cruising individuals' positions
    for i = 1:num_cruising
        % Generate a random direction
        direction = rand(1, num_variables) * 2 - 1;  % Random vector [-1, 1]
        
        % Normalize direction
        direction = direction / norm(direction);
        
        % Calculate cruising distance
        cruising_distance = cruising_distance * (1 - iter / max_iterations);  % Decrease cruising distance over time
        
        % Update position
        population(i, :) = population(i, :) + alpha * cruising_distance * direction;
        
        % Ensure the new position is within bounds
        population(i, :) = max(min(population(i, :), upper_limit), lower_limit);
    end
    
    % Update remaining individuals using random walk
    for i = num_cruising+1:population_size
        % Generate a random direction
        direction = rand(1, num_variables) * 2 - 1;  % Random vector [-1, 1]
        
        % Normalize direction
        direction = direction / norm(direction);
        
        % Update position
        population(i, :) = population(i, :) + alpha * direction;
        
        % Ensure the new position is within bounds
        population(i, :) = max(min(population(i, :), upper_limit), lower_limit);
    end
end

% Find the best solution
[best_fitness, best_index] = min(fitness);
best_solution = population(best_index, :);

% Display best solution found
fprintf('Best solution: %s\n', mat2str(best_solution));
fprintf('Best fitness: %f\n', best_fitness);

% Objective function (example)
function z = objective_function(x)
    % Example: Sphere function
    z = sum(x.^2);
end