% Fireworks Algorithm (FWA) Implementation with Sphere Function

% Define problem parameters
numParticles = 50; % Number of particles (fireworks)
numDimensions = 2; % Number of dimensions
numGenerations = 100; % Number of generations
alpha = 0.1; % Explosion amplitude factor
beta = 1; % Explosion magnitude scaling factor
delta_t = 1; % Time step
lowerLimit = -5; % Lower limit for each dimension
upperLimit = 5; % Upper limit for each dimension

% Initialize particle positions (random)
particles = rand(numParticles, numDimensions) * (upperLimit - lowerLimit) + lowerLimit;

% Main optimization loop
for t = 1:numGenerations
    % Step 1: Evaluate the fitness of each particle using the sphere function
    fitness = evaluateFitness(particles);
    
    % Step 2: Select the best particle (firework)
    [bestFitness, bestIndex] = min(fitness);
    bestParticle = particles(bestIndex, :);
    
    % Step 3: Generate sparks around the best particle
    sparks = generateSparks(bestParticle, alpha, beta, delta_t);
    
    % Step 4: Update the position of each particle
    particles = updateParticles(particles, sparks, lowerLimit, upperLimit);
    
    % Display best fitness in current generation
    disp(['Generation ', num2str(t), ': Best Fitness = ', num2str(bestFitness)]);
end

% Evaluate fitness function (Sphere function)
function fitness = evaluateFitness(x)
    fitness = sum(x.^2, 2);
end

% Generate sparks around the best particle
function sparks = generateSparks(bestParticle, alpha, beta, delta_t)
    numSparks = poissrnd(beta); % Poisson distribution for the number of sparks
    sparks = repmat(bestParticle, numSparks, 1) + alpha * randn(numSparks, numel(bestParticle)) * delta_t;
end

% Update the position of each particle
function newParticles = updateParticles(particles, sparks, lowerLimit, upperLimit)
    % Concatenate particles and sparks
    allParticles = [particles; sparks];
    
    % Apply boundary conditions
    allParticles = max(min(allParticles, upperLimit), lowerLimit);
    
    % Sort particles based on fitness
    fitness = evaluateFitness(allParticles);
    [~, sortedIndices] = sort(fitness);
    allParticles = allParticles(sortedIndices, :);
    
    % Select the top particles
    newParticles = allParticles(1:size(particles, 1), :);
end
