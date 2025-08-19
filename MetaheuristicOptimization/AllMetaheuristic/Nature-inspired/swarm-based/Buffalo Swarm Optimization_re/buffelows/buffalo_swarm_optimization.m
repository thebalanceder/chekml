% Buffalo Swarm Optimization (BSO) for Sphere Function

function [best_solution, best_fitness] = buffalo_swarm_optimization(num_buffaloes, num_iterations, num_variables, lower_bound, upper_bound)
    % Initialize buffalo population
    buffaloes = rand(num_buffaloes, num_variables) .* (upper_bound - lower_bound) + lower_bound;
    
    % Define objective function (sphere function)
    objective_function = @(x) sum((max(min(x, upper_bound), lower_bound)).^2); % Sphere function with bounds
    
    % Initialize best solution and best fitness
    best_solution = buffaloes(1,:);
    best_fitness = objective_function(best_solution);
    
    % Main loop
    for iter = 1:num_iterations
        % Update each buffalo's position
        for i = 1:num_buffaloes
            % Calculate new position based on local search
            new_position = local_search_with_bounds(buffaloes(i,:), objective_function, lower_bound, upper_bound);
            
            % Update buffalo position
            buffaloes(i,:) = new_position;
            
            % Update best solution and best fitness if needed
            fitness = objective_function(new_position);
            if fitness < best_fitness
                best_fitness = fitness;
                best_solution = new_position;
            end
        end
    end
end

% Local search function with bounds
function new_position = local_search_with_bounds(current_position, objective_function, lower_bound, upper_bound)
    % Perturb current position
    perturbation = randn(size(current_position)) * 0.1; % Small random perturbation
    new_position = current_position + perturbation;
    
    % Ensure new position is within bounds
    new_position = max(min(new_position, upper_bound), lower_bound);
end
