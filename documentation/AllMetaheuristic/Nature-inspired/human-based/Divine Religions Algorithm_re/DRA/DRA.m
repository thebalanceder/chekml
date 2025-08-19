
%_____________________________________________________________________________________________%
%  source code:  Divine Religions Algorithm (DRA)                                             %
%                                                                                             %
%  Developed in: MATLAB                                                                       %
% --------------------------------------------------------------------------------------------%
%  Main paper:   Divine Religions Algorithm: a novel social-inspired metaheuristic            %
%                algorithm for engineering and continuous optimization problems               %
%                 DOI: https://doi.org/10.1007/s10586-024-04954-x                             %                                                                 
%  Emails:       nima.khodadadi@miami.edu                                                     %
%_____________________________________________________________________________________________%
% Note:
% Due to the stochastic nature of metaheuristc algorithms, different runs
% may lead to slightly different results.
% -------------------------------------------------------------------------


function [BestCost,BestSol,cg_curve]=DRA(N,Max_Iteration,lb,ub,dim,func)%% Problem Definition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
%Function_name=sprintf('F%d',Benchmark_Function_ID);
format shortG
%[lb,ub,dim,func]=Get_Functions_details(Function_name);
fobj=func;                                   % Cost Function
NumberVariables=dim;                         % Number of Deciison Variables
VariablesSize=[1 NumberVariables];           % Decision Variables Matrix Size
VariablesMin=lb;                             % Decision Variables Lower Bound
VariablesMax=ub;                             % Decision Variables Upper Bound

%% Divine Religions Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MaxIteration=Max_Iteration;                  % Maximum Number of Iterations
BPsize=N;                                    % Belief Profile Size (Population Size)
NumberGroup=5;                               % Number of Groups
NumberFollower=BPsize-NumberGroup;           % Number of Followers
BPSP=0.5;                                    % Belief Profile Consideration Rate
MP=0.5;                                      % Miracle Rate
PP=0.9;                                      % proselytism Consideration Rate
RP=0.2;                                      % Reward or Penalty Consideration Rate
tic;
%% Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Empty Belief Profile Structure
empty_Belief.Belief=[];
empty_Belief.Cost=[];

% Initialize Belief Profile
BP=repmat(empty_Belief,BPsize,1);
Leader=repmat(empty_Belief,1,1);

% Create Initial Followers
for i=1:BPsize
    BP(i).Belief=unifrnd(VariablesMin,VariablesMax,VariablesSize);
    BP(i).Cost=fobj(BP(i).Belief);
end

% Sort Belief Profile
[~, SortOrder]=sort([BP.Cost]);
BP=BP(SortOrder);

%% Initialize Groups and Assign Missionarys & Followers %%%%%%%%%%%%%%%%%%%
missionary=BP(1:NumberGroup);
Follower=BP(NumberGroup+1:end);

empty_Group.missionary=[];
empty_Group.Follower=repmat(empty_Belief,0,1);
empty_Group.NumberFollower=0;
empty_Group.TotalCost=[];

Group=repmat(empty_Group,NumberGroup,1);

% Assign Missionaries to Groups
for k=1:NumberGroup
    Group(k).missionary=missionary(k);
end

% Assign Followers to Groups
for j=1:NumberFollower
    k=randi(NumberGroup);
    Group(k).Follower=[Group(k).Follower
            Follower(j)];
    Group(k).NumberFollower=Group(k).NumberFollower+1;
end
%% DRA Main Loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 for it=1:MaxIteration
    MP=(1*rand)*(1-(it/MaxIteration*2))*(1*rand());  %1-0
    
    [~, MinOrder]=min([BP.Cost]);           % Define a perfect Follower(Leader)
    Leader=BP(MinOrder);
    
    NewFollower=repmat(empty_Belief,1,1);   % Absorption an Follower and  Initialize Array for New Follower
    NewFollower.Belief=unifrnd(VariablesMin,VariablesMax,VariablesSize);
    NewFollower.Cost=fobj(NewFollower.Belief);
  
    if rand<=BPSP                           % Select Operator and Use Belief Profile
        NewFollower.Belief(randi([1 NumberVariables]))= BP(randi([1 BPsize])).Belief(randi([1 NumberVariables]));
    end
    
    % Exploration:
     if rand<=MP                            % Miracle Operator to Followers
         for i=1:BPsize  
            if rand<=0.5 
                BP(i).Belief=BP(i).Belief*cos(pi/2)*(rand-cos(rand));
            else
                BP(i).Belief = BP(i).Belief+rand*(BP(i).Belief-round(1^rand)* BP(i).Belief);
            end
            BP(i).Belief = max(BP(i).Belief,VariablesMin);
            BP(i).Belief = min(BP(i).Belief,VariablesMax);
    
            NewFobj=fobj(BP(i).Belief);
            if NewFobj <  BP(i).Cost
                BP(i).Cost=NewFobj;
            end
          end 
    % Exploitation:
     else                                   % proselytism Operator and Use Leader Profile
        NewFollower.Belief=Leader.Belief*(rand-sin(rand));   
        for i=1:BPsize
            if rand>(1-MP)
              BP(i).Belief = (BP(i).Belief*0.01) + (mean([BP(i).Belief])* (1-MP) + (1-(mean([BP(i).Belief])))-(rand-4*sin(sin(3.14*rand))));
            else
              BP(i).Belief= Leader.Belief*(rand-cos(rand));
            end  
            %BP(i).Belief = boundaryCheck(BP(i).Belief,VariablesMin,VariablesMax);
            BP(i).Belief = max(BP(i).Belief,VariablesMin);
            BP(i).Belief = min(BP(i).Belief,VariablesMax);
    
            NewFobj=fobj(BP(i).Belief);
            if NewFobj <  BP(i).Cost
                BP(i).Cost=NewFobj;
            end
        end
    end
%% Calculating Fitness New Follower %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %NewFollower.Belief = boundaryCheck(NewFollower.Belief,VariablesMin,VariablesMax);
    NewFollower.Belief = max(NewFollower.Belief,VariablesMin);
    NewFollower.Belief = min(NewFollower.Belief,VariablesMax);
    
    NewFobj=fobj(NewFollower.Belief);
    if NewFobj <  BP(i).Cost
        NewFollower.Cost=NewFobj;
    end
%% Reward or penalty operator %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    index=randi([1 BPsize]);
    if rand>=RP
        BP(index).Belief=BP(index).Belief * (1-randn());        % Reward Operator for Followers
        %BP(index).Belief = boundaryCheck(BP(index).Belief,VariablesMin,VariablesMax);
        BP(index).Belief = max(BP(index).Belief,VariablesMin);
        BP(index).Belief = min(BP(index).Belief,VariablesMax);
    
        NewFobj=fobj(BP(index).Belief);
        if NewFobj <  BP(i).Cost
            BP(i).Cost=NewFobj;
        end
    else
        BP(index).Belief=BP(index).Belief * (1+randn());        % penalty Operator for Followers
        %BP(index).Belief = boundaryCheck(BP(index).Belief,VariablesMin,VariablesMax);
        BP(index).Belief = max(BP(index).Belief,VariablesMin);
        BP(index).Belief = min(BP(index).Belief,VariablesMax);
        NewFobj=fobj(BP(index).Belief);
        if NewFobj <  BP(i).Cost
            BP(i).Cost=NewFobj;
        end
    end      
%% Update Belief Profile in other word Merge Belief Profile and New Follower
    if NewFobj <  BP(end).Cost
        BP(end)=NewFollower;
    end

%% Replacement operator %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    missionary=Group(k).missionary;
    Follower=Group(k).Follower(end);
            
    Group(k).missionary=Follower;
    Group(k).Follower(end)=missionary;
    
%% Define Results%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Store Best Cost Ever Found
    BestSol=BP(1).Cost;
    BestCost=BP(1).Cost;
    BestCost(it)=BP.Cost;
    cg_curve(it)=BP.Cost;

 
end