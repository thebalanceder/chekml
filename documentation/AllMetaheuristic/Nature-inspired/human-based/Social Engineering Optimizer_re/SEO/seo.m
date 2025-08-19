function [best_solution, best_fitness] = seo(obj_func, num_variables, num_agents, max_iter, lb, ub)
    % Initialize population
    population = bsxfun(@plus, lb, bsxfun(@times, rand(num_agents, num_variables), (ub - lb)));
    fitness = zeros(1, num_agents);

    % Evaluate fitness of each agent
    for i = 1:num_agents
        fitness(i) = obj_func(population(i, :));
    end

    % Main loop
    for iter = 1:max_iter
        % Update position and fitness of each agent
        for i = 1:num_agents
            % Select a random agent as the target
            target_index = randi(num_agents);
            while target_index == i
                target_index = randi(num_agents);
            end
            
            % Socially engineer the agent's solution based on the target
            new_solution = population(i, :) + randn(1, num_variables) .* (population(target_index, :) - population(i, :));
            
            % Clip new solution to ensure it stays within bounds
            new_solution = max(min(new_solution, ub), lb);
            
            % Evaluate fitness of the new solution
            new_fitness = obj_func(new_solution);
            
            % Update if the new solution is better
            if new_fitness < fitness(i)
                population(i, :) = new_solution;
                fitness(i) = new_fitness;
            end
        end
    end

    % Find the best solution in the final population
    [best_fitness, best_index] = min(fitness);
    best_solution = population(best_index, :);
end
