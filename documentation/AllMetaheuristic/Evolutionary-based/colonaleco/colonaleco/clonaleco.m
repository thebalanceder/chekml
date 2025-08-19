% Constants
num_generators = 3; % Number of generators
load_demand = 300; % Load demand
max_capacity = [200, 150, 100]; % Maximum generation capacity for each generator
min_capacity = [50, 30, 20]; % Minimum generation capacity for each generator
penalty_factor = 1000; % Penalty factor for violating constraints
population_size = 100; % Size of antibody population
max_iterations = 100; % Maximum number of iterations
mutation_rate = 0.1; % Mutation rate
ccost=[561 7.92 0.00156  600 150 300 0.0315
       310 7.85 0.00194  400 100 200 0.042
       78  7.97 0.00482  200 50  150 0.063]; 
VarLow=[150 100 50];
VarHigh =[600 400 200];
% Define quadratic cost coefficients for each generator
a = [0.01, 0.015, 0.02]; % Quadratic coefficient
b = [0.5, 0.75, 1]; % Linear coefficient
c = [10, 15, 20]; % Constant coefficient

% Initialize population randomly within the feasible region
population = rand(population_size, num_generators) .* (max_capacity - min_capacity) + min_capacity;

% Main loop
for iter = 1:max_iterations
    % Evaluate fitness of antibodies
    fitness = evaluate_fitness(population, load_demand, penalty_factor, min_capacity, max_capacity, a, b, c);
    
    % Clonal Selection
    [cloned_population, cloned_fitness] = clonal_selection(population, fitness);
    
    % Mutate cloned antibodies
    mutated_population = mutate(cloned_population, mutation_rate, min_capacity, max_capacity);
    
    % Evaluate fitness of mutated antibodies
    mutated_fitness = evaluate_fitness(mutated_population, load_demand, penalty_factor, min_capacity, max_capacity, a, b, c);
    
    % Select antibodies for the next generation
    population = select_population(population, fitness, mutated_population, mutated_fitness);
end

% Obtain the best solution
best_solution = select_best_solution(population, fitness);

disp('Best solution:');
disp(best_solution); 
disp(sum(best_solution)); 
cost1=objective_function(best_solution,load_demand, a, b, c);
disp(cost1);
% Define objective function 
function cost = objective_function(generation, load_demand, a, b, c)
    cost = sum(a .* generation.^2 + b .* generation + c); % Quadratic cost function
   % penalty = max(0, sum(generation) - load_demand); % Penalty for violating load demand constraint
     penalty = 1000*abs(sum(generation) - load_demand); % Penalty for violating load d
    cost = cost + penalty;
end

% Evaluate fitness of antibodies
function fitness = evaluate_fitness(population, load_demand, penalty_factor, min_capacity, max_capacity, a, b, c)
    num_population = size(population, 1);
    fitness = zeros(num_population, 1);
    for i = 1:num_population
        if all(population(i, :) >= min_capacity) && all(population(i, :) <= max_capacity) && sum(population(i, :)) >= load_demand
            fitness(i) = objective_function(population(i, :), load_demand, a, b, c);
        else
            fitness(i) = inf; % Assign infinite fitness to infeasible solutions
        end
    end
end

% Clonal Selection
function [cloned_population, cloned_fitness] = clonal_selection(population, fitness)
    num_population = size(population, 1);
    [~, sorted_indices] = sort(fitness);
    cloned_population = [];
    cloned_fitness = [];
    for i = 1:num_population
        num_clones = ceil(num_population / (i + 1));
        for j = 1:num_clones
            cloned_population = [cloned_population; population(sorted_indices(i), :)];
            cloned_fitness = [cloned_fitness; fitness(sorted_indices(i))];
        end
    end
end

% Mutate cloned antibodies
function mutated_population = mutate(cloned_population, mutation_rate, min_capacity, max_capacity)
    num_clones = size(cloned_population, 1);
    num_genes = size(cloned_population, 2);
    mutated_population = cloned_population;
    for i = 1:num_clones
        for j = 1:num_genes
            if rand < mutation_rate
                mutated_population(i, j) = min(max_capacity(j), max(min_capacity(j), cloned_population(i, j) + randn));
            end
        end
    end
end

% Select antibodies for the next generation
function population = select_population(population, fitness, mutated_population, mutated_fitness)
    total_population = [population; mutated_population];
    total_fitness = [fitness; mutated_fitness];
    [~, sorted_indices] = sort(total_fitness);
    population = total_population(sorted_indices(1:length(population)), :);
end

% Select the best solution
function best_solution = select_best_solution(population, fitness)
    [~, best_index] = min(fitness);
    best_solution = population(best_index, :);
end
