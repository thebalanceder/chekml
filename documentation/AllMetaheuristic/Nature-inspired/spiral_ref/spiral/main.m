% Define the Sphere function
sphereFunc = @(x) sum(x.^2);

% Define problem dimensionality
dim = 2;

% Define lower and upper bounds for each dimension
lb = -10 * ones(1, dim);
ub = 10 * ones(1, dim);

% Set algorithm parameters
maxIter = 100;  % Maximum number of iterations
nPop = 50;      % Population size
c = 0.1;        % Spiral coefficient
alpha = 1;      % Not used in this implementation

% Run Spiral Optimization algorithm
[bestSolution, bestFitness] = SPO(sphereFunc, dim, lb, ub, maxIter, nPop, c, alpha);

disp('Best Solution:');
disp(bestSolution);
disp('Best Fitness:');
disp(bestFitness);
