% Giza Pyramids Construction (GPC) Algorithm
%
% Author : Sasan Harifi
%
% Paper  : Giza Pyramids Construction: an ancient-inspired metaheuristic algorithm for optimization
% DOI    : http://dx.doi.org/10.1007/s12065-020-00451-3
%
% Copyright (c) 2020, All rights reserved.
% Please read the "license.txt" for license terms.
%
% Code Publisher: http://www.harifi.com
% -------------------------------------------------
% This demo only implements a standard version of GPC for minimization of
% a standard test function (Sphere) on MATLAB R2015b.
% -------------------------------------------------
% Note:
% Due to the stochastic nature of metaheuristc algorithms, different runs
% may lead to slightly different results.
% -------------------------------------------------

clc;
clear;
close all;

%% Problem Definition
CostFunction=@(x) Sphere(x);        % Cost Function

nVar=30;                  % Number of Decision Variables

VarSize=[1 nVar];         % Decision Variables Matrix Size

VarMin=-5.12;             % Decision Variables Lower Bound
VarMax= 5.12;             % Decision Variables Upper Bound

%% Giza Pyramids Construction (GPC) Parameters

MaxIteration=1000;   % Maximum Number of Iterations (Days of work)

nPop=20;             % Number of workers

G = 9.8;             % Gravity
Tetha = 14;          % Angle of Ramp
MuMin = 1;           % Minimum Friction 
MuMax = 10;          % Maximum Friction
pSS= 0.5;            % Substitution Probability

%% Initialization
% Empty Stones Structure
stone.Position=[];
stone.Cost=[];

% Initialize Population Array
pop=repmat(stone,nPop,1);

% Initialize Best Solution Ever Found
best_worker.Cost=inf;

% Create Initial Stones
for i=1:nPop
   pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
   pop(i).Cost=CostFunction(pop(i).Position);
   if pop(i).Cost<=best_worker.Cost
       best_worker=pop(i);          % as Pharaoh's special agent
   end
end

% Array to Hold Best Cost Values
BestCost=zeros(MaxIteration,1);

%% Giza Pyramids Construction (GPC) Algorithm Main Loop
for it=1:MaxIteration
    newpop=repmat(stone,nPop,1);
    
    for i=1:nPop
        newpop(i).Cost = inf;
       
        V0= rand(1,1);                          % Initial Velocity                                      
        Mu= MuMin+(MuMax-MuMin)*rand(1,1);      % Friction

        d = (V0^2)/((2*G)*(sind(Tetha)+(Mu*cosd(Tetha))));                  % Stone Destination
        x = (V0^2)/((2*G)*(sind(Tetha)));                                   % Worker Movement
        epsilon=unifrnd(-((VarMax-VarMin)/2),((VarMax-VarMin)/2),VarSize);  % Epsilon
        newsol.Position = (pop(i).Position+d).*(x*epsilon);                 % Position of Stone and Worker
      % newsol.Position = (pop(i).Position+d)+(x*epsilon);                  % Note: In some cases or some problems use this instead of the previous line to get better results

        newsol.Position=max(newsol.Position,VarMin);
        newsol.Position=min(newsol.Position,VarMax);
        
        % Substitution
        z=zeros(size(pop(i).Position));
        k0=randi([1 numel(pop(i).Position)]);
        for k=1:numel(pop(i).Position)
            if k==k0 || rand<=pSS
                z(k)=newsol.Position(k);
            else
                z(k)=pop(i).Position(k);
            end
        end
        
        newsol.Position=z;
        newsol.Cost=CostFunction(newsol.Position);
        
        if newsol.Cost <= newpop(i).Cost
           newpop(i) = newsol;
           if newpop(i).Cost<=best_worker.Cost
               best_worker=newpop(i);
           end
        end
    
    end
      
    % Merge
    pop=[pop 
         newpop];  %#ok
    
    % Sort
    [~, SortOrder]=sort([pop.Cost]);
    pop=pop(SortOrder);
    
    % Truncate
    pop=pop(1:nPop);

    % Store Best Cost Ever Found
    BestCost(it)=pop(1).Cost;
    
    % Show Iteration Information
    disp(['It:' num2str(it) ', Cost => ' num2str(BestCost(it))]);
end

figure;
%plot(BestCost,'LineWidth',2);
semilogy(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
%grid on;