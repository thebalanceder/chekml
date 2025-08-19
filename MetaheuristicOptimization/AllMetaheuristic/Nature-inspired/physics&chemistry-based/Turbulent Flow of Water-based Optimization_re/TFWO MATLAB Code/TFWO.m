
clear all
clc
disp('Turbulent Flow of Waterbased Optimization (TFWO)');


%% Problem Definition

CostFunction=@(x) Cost(x);

nVar=30;       % Number of Decision Variables
VarMin(1,1:nVar)=-100;   
VarMax(1,1:nVar)= -VarMin;
 %% TFWO Settings

nWh=3; % Number of Whirlpools
 nObW=30; % Number of Particles(Objects) for each Whirlpool   
   nPop=nWh+nWh*nObW;  % Toatl Number of population
nOb=nPop-nWh; % Toatl Number of Particles(Objects)

 MaxIter=3000;

%% Initialization :
% 1-Generate the initial random population
% 2-Evaluate the fitness the initial population
% 3-The divide the algorithm痴 population into NWh groups or whirlpool sets

ShareSettings;

Whirlpool=Initialize();

BestSol.Position=[];
BestSol.Cost=[];

BestCost=zeros(TFWOSettings.MaxIter,1);
MeanCost=zeros(TFWOSettings.MaxIter,1);

%% TFWO
for iter=1:MaxIter
    Whirlpool=Effectsofwhirlpools(Whirlpool, iter);%Pseudocodes From 1 To 5    
    Whirlpool=Pseudocode6(Whirlpool, iter); %Pseudocode 6   
    WhirlpoolCost=[Whirlpool.Cost];
    [BestWhirlpoolCost BestWhirlpoolIndex]=min(WhirlpoolCost);
    BestWhirlpool=Whirlpool(BestWhirlpoolIndex);
    
    BestSol.Position=BestWhirlpool.Position;
    BestSol.Cost=BestWhirlpool.Cost;

    S(iter,:)=BestSol.Position;
    BestCost(iter)=BestWhirlpoolCost;
    MeanCost(iter)=mean(WhirlpoolCost);
    
    disp(['Iter ' num2str(iter) ...
          ': Best Cost = ' num2str(BestCost(iter)) ...
          ]);
       
end

 hold on
plot(log(BestCost),'m--','LineWidth',3); 
 Cost_Rsult=BestCost(end);
Rsult=BestSol.Position

