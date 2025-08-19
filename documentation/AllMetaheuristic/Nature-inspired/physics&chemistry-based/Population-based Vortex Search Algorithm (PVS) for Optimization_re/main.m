%{
%% Please refer to the main paper: (Published: 18 August 2022) 
  TITLE : PVS: a new population-based vortex search algorithm with boosted exploration capability using polynomial mutation
 AUTHOR : Tahir SAG (Department of Computer Engineering, Selcuk University, Turkey)
JOURNAL : Neural Computing and Applications, Volume:34, Pages:18211–18287 (2022). 
    DOI : https://doi.org/10.1007/s00521-022-07671-x
          https://link.springer.com/article/10.1007/s00521-022-07671-x
  email : tahirsag@selcuk.edu.tr
%}

%% PVS: Population-based Vortex Search Algorithm
clc
clear
close all
rng('shuffle');

%% Function parameters
objfun = 'sphere';
dim = 30;
lb = -100;
ub = 100;
opt_f = 0.0;
err = 0; % err : admissible error

%% Algorithm parameters
popsize = 50;
maxFEs = dim * 10000;

%% Run PVS
tic
[gbest, gmin, max_FEs, iter_results] =...
    PVS(popsize, dim, maxFEs, objfun, err, lb, ub, opt_f);
elapsedTime = toc;

fprintf(['The best solution obtained by PVS for %s function\n' ...
         'Objective Value = %g\n' ...
         '   Elapsed Time = %g seconds.\n'], upper(objfun), gmin, elapsedTime);

