%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Partial Reinforcement Optimizer: An Evolutionary Optimization Algorithm %
% version 1.0                                                             %
% Authiors:                                                               %
% Ahmad Taheri **, Keyvan RahimiZadeh, Amin Beheshti, Jan Baumbach,       %
% Ravipudi Venkata Rao, Seyedali Mirjalili, Amir H Gandomi                %
%                                                                         %
% ** E-mail:                                                              %
%          Ahmad.taheri@uni-hamburg.de                                    %
%          Ahmad.thr@gmail.com                                            %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
% -------------------------------------------------------------------------
%       **********  This Code runs the PRO algorithm  ***********
% -------------------------------------------------------------------------

%% --------------------    settings    -------------------------------- %%
% This section is to initialize Parameters
%%--------------------------------------------------------------------- %%
   Func_id = 1;          % CEC2017 1~30
   nPop = 30;            % Population Size
   N = 10;               % number of Decision variables 
   MaxFEs = N * 10000;   % Maximum number of function evaluations
   NumofExper = 1;       % Number of test
  %Benchmark = 3;        % 1:Classic 2:CEC2005 3:CEC2014 
   LB=-100;%lb;          % Lower Bound
   UB=100;%ub;           % Upper Bound
% =====================================================================================
%for Func_id = 1:1
    
%clear;
clc;

global initial_flag
initial_flag = 0;
%% 
Function_name=['F' num2str(Func_id)];
%========== CEC2017 ==========
%fhd=str2func('cec14_func');
fhd=str2func('cec17_func');
CostFunction=Func_id;
%============================= 
LB = LB.*ones(1,N);       
UB = UB.*ones(1,N);       

% Empty Solution Structure
empty_Solution.Position=[];
empty_Solution.Cost=[];
 
Population=repmat(empty_Solution,nPop,1);
SumBestCostPRO_=zeros(MaxFEs,1);
BestSolCostPRO= []; %zeros(MaxFEs,1);

%===================================================

for ii=1:NumofExper
    
  rand('state',sum(100*clock));
  initial_flag = 0; % should set the flag to 0 for each run, each function
  
   % Create Initial Population
for i=1:nPop
    Population(i).Position= LB+rand(1,N).*(UB-LB); %LB+rand(1,N).*(UB-LB);    
    %Population(i).Cost = feval(fhd,Population(i).Position',CostFunction) -  (CostFunction*100); % CEC2014 F(X) - F(X*)
    Population(i).Cost = feval(fhd,Population(i).Position',CostFunction) ; % CEC2017 
end  
    
   %% 
%tic;

% --------  Call PRO algorithm to optimize the selected function --------%%
[BestCostPRO_,BestSolCostPRO(ii)]=PRO_v1(N,MaxFEs,LB,UB,Population,nPop,CostFunction); 

disp([num2str(ii),' BestCost: ', num2str(BestSolCostPRO(ii))]);

SumBestCostPRO_=SumBestCostPRO_+ BestCostPRO_(1:MaxFEs);
%T2 = toc;
end


AveBestCostPRO_=SumBestCostPRO_ ./ NumofExper;

Mean = mean(BestSolCostPRO);
SD   = std(BestSolCostPRO);

%% -----------------        Save the Results        -------------------- %%
%% --------------------------------------------------------------------- %%
filename=['PRO Result CEC17 D10_NFE100K_10t _' Function_name '.mat']
save(filename);

%% ----------------- Visualize the Convergence Rate -------------------- %%
%% --------------------------------------------------------------------- %%
f1=figure;
semilogy(AveBestCostPRO_,'r-','LineWidth',2);
grid on;
hold off;
xlabel('FEs');
str=['F(x) = ' Function_name];
ylabel(str);
legend('PRO');

%end %for