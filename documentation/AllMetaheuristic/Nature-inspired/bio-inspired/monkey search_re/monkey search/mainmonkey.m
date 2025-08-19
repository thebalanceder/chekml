% Define the objective function (Sphere function)
objective_function = @(x) sum(x.^2);

% Set up the parameters
lb = -10; % Lower bound
ub = 10;  % Upper bound
population_size = 20; % Population size
max_iter = 100; % Maximum number of iterations

% Call the MSA algorithm
[best_solution, best_fitness] = MSA(objective_function, lb, ub, population_size, max_iter);

% Display the result
disp(['Best Solution: ', num2str(best_solution)]);
disp(['Best Fitness: ', num2str(best_fitness)]);
function [best_solution, best_fitness] = MSA(objective_function, lb, ub, population_size, max_iter)
    % Initialize the population
    population = rand(population_size, numel(lb)) .* (ub - lb) + lb;
    
    % Evaluate the fitness of the population
    fitness = zeros(population_size, 1);
    for i = 1:population_size
        fitness(i) = objective_function(population(i, :));
    end
    
    % Find the best solution and fitness
    [best_fitness, idx] = min(fitness);
    best_solution = population(idx, :);
    
    % Main loop
    for iter = 1:max_iter
        % Generate a random number for each dimension
        r = rand(population_size, numel(lb));
        
        % Update individuals' positions
        for i = 1:population_size
            for j = 1:numel(lb)
                if r(i, j) < 0.5
                    population(i, j) = best_solution(j) + rand() * (ub(j) - best_solution(j));
                else
                    population(i, j) = best_solution(j) - rand() * (best_solution(j) - lb(j));
                end
                
                % Bound the positions
                population(i, j) = max(population(i, j), lb(j));
                population(i, j) = min(population(i, j), ub(j));
            end
            
            % Evaluate the fitness of the updated individual
            fitness(i) = objective_function(population(i, :));
        end
        
        % Update the best solution and fitness
        [current_best_fitness, idx] = min(fitness);
        if current_best_fitness < best_fitness
            best_fitness = current_best_fitness;
            best_solution = population(idx, :);
        end
        
        % Display the iteration information
        disp(['Iteration ', num2str(iter), ': Best Fitness = ', num2str(best_fitness)]);
    end
end
