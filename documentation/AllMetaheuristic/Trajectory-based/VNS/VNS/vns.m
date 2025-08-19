
% Parameters
num_neighborhoods = 5;          % Number of neighborhoods
neighborhood_sizes = [1, 2, 3]; % Size of each neighborhood
num_variables = 10;             % Number of variables
max_iterations = 100;           % Maximum number of iterations
mutation_rate = 0.1;            % Mutation rate
lower_limit = -5;               % Lower limit for variables
upper_limit = 5;                % Upper limit for variables

% Initialization
current_solution = lower_limit + (upper_limit - lower_limit) * rand(1, num_variables); % Random initial solution
current_solution = max(min(current_solution, upper_limit), lower_limit); % Ensure initial solution is within bounds
current_fitness = objective_function(current_solution);

% Main loop
for iter = 1:max_iterations
    % Randomly select a neighborhood
    neighborhood_index = randi(num_neighborhoods);
    neighborhood_size = neighborhood_sizes(neighborhood_index);
    
    % Generate a random neighbor within the selected neighborhood
    neighbor = current_solution + mutation_rate * (rand(1, num_variables) - 0.5) * neighborhood_size;
    
    % Ensure the neighbor is within bounds
    neighbor = max(min(neighbor, upper_limit), lower_limit);
    
    % Evaluate the neighbor
    neighbor_fitness = objective_function(neighbor);
    
    % Update current solution if the neighbor is better
    if neighbor_fitness < current_fitness
        current_solution = neighbor;
        current_fitness = neighbor_fitness;
    end
end

% Display best solution found
fprintf('Best solution: %s\n', mat2str(current_solution));
fprintf('Best fitness: %f\n', current_fitness);
% Variable Neighborhood Search (VNS) Implementation

% Objective function (example)
function z = objective_function(x)
    % Example: Sphere function
    z = sum(x.^2);
end
