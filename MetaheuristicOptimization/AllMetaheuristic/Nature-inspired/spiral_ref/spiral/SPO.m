function [bestSolution, bestFitness] = SPO(objFunc, dim, lb, ub, maxIter, nPop, c, alpha)
    % Initialize population
    population = repmat(lb, nPop, 1) + rand(nPop, dim) .* repmat(ub - lb, nPop, 1);
    fitness = zeros(nPop, 1);
    
    % Evaluate initial population
    for i = 1:nPop
        fitness(i) = objFunc(population(i, :));
    end
    
    % Initialize best solution and fitness
    [bestFitness, bestIndex] = min(fitness);
    bestSolution = population(bestIndex, :);
    
    % Main loop
    for iter = 1:maxIter
        % Generate spiral movement
        r = rand(nPop, 1); % Generate random numbers between 0 and 1
        theta = 2 * pi * rand(nPop, 1); % Generate random angles between 0 and 2*pi
        
        % Calculate new positions
        newX = population(:, 1) + c * r .* cos(theta);
        newY = population(:, 2) + c * r .* sin(theta);
        
        % Ensure new positions are within bounds
        newX = min(ub(1), max(lb(1), newX));
        newY = min(ub(2), max(lb(2), newY));
        
        % Combine new positions
        newPosition = [newX, newY];
        
        % Evaluate new solutions
        for i = 1:nPop
            fitness(i) = objFunc(newPosition(i, :));
        end
        
        % Update best solution
        [minFitness, minIndex] = min(fitness);
        if minFitness < bestFitness
            bestFitness = minFitness;
            bestSolution = newPosition(minIndex, :);
        end
        
        % Select the best solutions to survive
        [~, sortedIndices] = sort(fitness);
        population = newPosition(sortedIndices(1:nPop), :);
    end
end
