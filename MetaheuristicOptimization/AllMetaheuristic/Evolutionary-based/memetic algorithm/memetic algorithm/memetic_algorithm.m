function [best_solution, best_fitness] = memetic_algorithm(obj_func, dim, lower_bounds, upper_bounds, max_iter, pop_size, mutation_rate, crossover_rate)
    % Initialize population
    population = lower_bounds + (upper_bounds - lower_bounds) .* rand(pop_size, dim);
    
    % Evaluate initial population
    fitness = zeros(pop_size, 1);
    for i = 1:pop_size
        fitness(i) = obj_func(population(i, :));
    end
    
    % Main loop
    for iter = 1:max_iter
        % Perform local search (hill climbing) on a portion of the population
        local_search_pop_size = round(0.1 * pop_size); % adjust this fraction as needed
        for i = 1:local_search_pop_size
            % Perform hill climbing starting from a random individual
            individual = population(randi(pop_size), :);
            new_individual = hill_climbing(obj_func, individual, lower_bounds, upper_bounds);
            
            % Replace the individual if the new one is better
            new_fitness = obj_func(new_individual);
            if new_fitness < fitness(i)
                population(i, :) = new_individual;
                fitness(i) = new_fitness;
            end
        end
        
        % Select parents for crossover
        [~, sorted_indices] = sort(fitness);
        parent_indices = sorted_indices(1:round(crossover_rate * pop_size));
        
        % Perform crossover
        children = zeros(length(parent_indices), dim);
        for i = 1:2:length(parent_indices)
            parent1 = population(parent_indices(i), :);
            parent2 = population(parent_indices(i+1), :);
            children(i:i+1, :) = crossover(parent1, parent2);
        end
        
        % Perform mutation
        for i = 1:length(parent_indices)
            if rand < mutation_rate
                children(i, :) = mutation(children(i, :), lower_bounds, upper_bounds);
            end
        end
        
        % Replace worst individuals with offspring
        [~, worst_indices] = max(fitness);
        num_worst = length(worst_indices);
        population(worst_indices, :) = children(1:num_worst, :);
        
        % Update fitness values
        for i = 1:num_worst
            fitness(worst_indices(i)) = obj_func(population(worst_indices(i), :));
        end
    end
    
    % Find the best solution
    [best_fitness, best_index] = min(fitness);
    best_solution = population(best_index, :);
end

function new_solution = hill_climbing(obj_func, initial_solution, lower_bounds, upper_bounds)
    current_solution = initial_solution;
    current_fitness = obj_func(current_solution);
    
    % Perform hill climbing
    max_iter = 100; % adjust as needed
    for iter = 1:max_iter
        % Generate a new solution by perturbing the current one
        new_solution = current_solution + 0.1 * randn(size(current_solution));
        % Ensure the new solution is within bounds
        new_solution = max(lower_bounds, min(upper_bounds, new_solution));
        
        % If the new solution is better, replace the current one
        new_fitness = obj_func(new_solution);
        if new_fitness < current_fitness
            current_solution = new_solution;
            current_fitness = new_fitness;
        end
    end
    
    new_solution = current_solution;
end

function offspring = crossover(parent1, parent2)
    % Single point crossover
    crossover_point = randi([1, length(parent1)]);
    offspring1 = [parent1(1:crossover_point), parent2(crossover_point+1:end)];
    offspring2 = [parent2(1:crossover_point), parent1(crossover_point+1:end)];
    offspring = [offspring1; offspring2];
end

function mutated_solution = mutation(solution, lower_bounds, upper_bounds)
    % Random mutation
    mutation_range = 0.1; % adjust as needed
    mutated_solution = solution + mutation_range * randn(size(solution));
    % Ensure the mutated solution is within bounds
    mutated_solution = max(lower_bounds, min(upper_bounds, mutated_solution));
end
