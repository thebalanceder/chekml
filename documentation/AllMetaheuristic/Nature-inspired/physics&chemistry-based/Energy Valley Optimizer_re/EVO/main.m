% Parameters
num_particles = 50;             % Number of particles (drops of water)
num_dimensions = 2;             % Number of dimensions (variables)
max_iterations = 100;           % Maximum number of iterations
step_size = 0.1;                % Step size (velocity coefficient)
alpha = 0.9;                    % Momentum coefficient
beta = 0.2;                     % Learning rate
lb = -10;                       % Lower bounds for variables
ub = 10;                        % Upper bounds for variables

% Initialization: Randomly initialize particle positions and velocities
particles = rand(num_particles, num_dimensions) * (ub - lb) + lb;
velocity = rand(num_particles, num_dimensions) * (ub - lb) + lb;
fitness = zeros(num_particles, 1); % Fitness values

% Main loop (Energy Valley Optimizer)
for iteration = 1:max_iterations
    % Evaluate fitness for each particle
    for i = 1:num_particles
        fitness(i) = sphere_function(particles(i, :));
    end
    
    % Sort particles based on fitness
    [fitness, sorted_indices] = sort(fitness);
    particles = particles(sorted_indices, :);
    
    % Calculate gradient (approximation using finite differences)
    gradient = zeros(num_particles, num_dimensions);
    for i = 1:num_particles
        for j = 1:num_dimensions
            x_plus = particles(i, :);
            x_plus(j) = x_plus(j) + beta;
            x_minus = particles(i, :);
            x_minus(j) = x_minus(j) - beta;
            gradient(i, j) = (sphere_function(x_plus) - sphere_function(x_minus)) / (2 * beta);
        end
    end
    
    % Update particle positions and velocities
    velocity = alpha * velocity + step_size * gradient;
    particles = particles - velocity;
    
    % Ensure particles stay within bounds [lb, ub]
    particles = min(max(particles, lb), ub);
    
    % Display best solution
    best_solution = particles(1, :);
    best_fitness = fitness(1);
    fprintf('Iteration %d: Best Fitness = %f\n', iteration, best_fitness);
end

% Display final result
fprintf('Optimization finished.\n');
fprintf('Best solution found: %s\n', mat2str(best_solution));
fprintf('Best fitness: %f\n', best_fitness);