%% Imperialist Competitive Algorithm (CCA);

% A Socio Politically Inspired Optimization Strategy.
% 2008

% To use this code, you should only prepare your cost function and apply
% CCA to it. Please read the guide available in my home page. 

% Special thank is for my friend Mostapha Kalami Heris whos breadth wision toward
% artificial intelligence and his programming skils have always been a
% sourse of inspiration for me. He helped me a lot to prepare this code.
% His email: sm.kalami@gmail.com

% ----------------------------------------
% Esmaeil Atashpaz Gargari
% Control and Intelligent Processing Center of Excellence,
% ECE school of University of Tehran, Iran
% Cellphone: (+98)-932-9011620
% Email: e.atashpaz@ece.ut.ac.ir & atashpaz.gargari@gmail.com
% Home Page: http://www.atashpaz.com

close all
clc; clear
%% Problem Statement
ProblemParams.CostFuncName = 'BenchmarkFunction';    % You should state the name of your cost function here.
ProblemParams.CostFuncExtraParams = 6;
ProblemParams.NPar = 30;                           % Number of optimization variables of your objective function. "NPar" is the dimention of the optimization problem.
ProblemParams.VarMin = -6;                         % Lower limit of the optimization parameters. You can state the limit in two ways. 1)   2)
ProblemParams.VarMax = 6;                       % Lower limit of the optimization parameters. You can state the limit in two ways. 1)   2)

% Modifying the size of VarMin and VarMax to have a general form
if numel(ProblemParams.VarMin)==1
    ProblemParams.VarMin=repmat(ProblemParams.VarMin,1,ProblemParams.NPar);
    ProblemParams.VarMax=repmat(ProblemParams.VarMax,1,ProblemParams.NPar);
end

ProblemParams.SearchSpaceSize = ProblemParams.VarMax - ProblemParams.VarMin;

%% Algorithmic Parameter Setting
AlgorithmParams.NumOfCountries = 200;               % Number of initial countries.
AlgorithmParams.NumOfInitialImperialists = 8;      % Number of Initial Imperialists.
AlgorithmParams.NumOfAllColonies = AlgorithmParams.NumOfCountries - AlgorithmParams.NumOfInitialImperialists;
AlgorithmParams.NumOfDecades = 2000;
AlgorithmParams.RevolutionRate = 0.3;               % Revolution is the process in which the socio-political characteristics of a country change suddenly.
AlgorithmParams.AssimilationCoefficient = 2;        % In the original paper assimilation coefficient is shown by "beta".
AlgorithmParams.AssimilationAngleCoefficient = .5;  % In the original paper assimilation angle coefficient is shown by "gama".
AlgorithmParams.Zeta = 0.02;                        % Total Cost of Empire = Cost of Imperialist + Zeta * mean(Cost of All Colonies);
AlgorithmParams.DampRatio = 0.99;
AlgorithmParams.StopIfJustOneEmpire = false;         % Use "true" to stop the algorithm when just one empire is remaining. Use "false" to continue the algorithm.
AlgorithmParams.UnitingThreshold = 0.02;            % The percent of Search Space Size, which enables the uniting process of two Empires.

zarib = 1.05;                       % **** Zarib is used to prevent the weakest impire to have a probability equal to zero
alpha = 0.1;                        % **** alpha is a number in the interval of [0 1] but alpha<<1. alpha denotes the importance of mean minimum compare to the global mimimum.

%% Display Setting
DisplayParams.PlotEmpires = false;    % "true" to plot. "false" to cancel ploting.
if DisplayParams.PlotEmpires
    DisplayParams.EmpiresFigureHandle = figure('Name','Plot of Empires','NumberTitle','off');
    DisplayParams.EmpiresAxisHandle = axes;
end

DisplayParams.PlotCost = true;    % "true" to plot. "false"
if DisplayParams.PlotCost
    DisplayParams.CostFigureHandle = figure('Name','Plot of Minimum and Mean Costs','NumberTitle','off');
    DisplayParams.CostAxisHandle = axes; 
end

ColorMatrix = [1   0   0  ; 0 1   0    ; 0   0 1    ; 1   1   0  ; 1   0 1    ; 0 1   1    ; 1 1 1       ;
               0.5 0.5 0.5; 0 0.5 0.5  ; 0.5 0 0.5  ; 0.5 0.5 0  ; 0.5 0 0    ; 0 0.5 0    ; 0 0 0.5     ;
               1   0.5 1  ; 0.1*[1 1 1]; 0.2*[1 1 1]; 0.3*[1 1 1]; 0.4*[1 1 1]; 0.5*[1 1 1]; 0.6*[1 1 1]];
DisplayParams.ColorMatrix = [ColorMatrix ; sqrt(ColorMatrix)];

DisplayParams.AxisMargin.Min = ProblemParams.VarMin;
DisplayParams.AxisMargin.Max = ProblemParams.VarMax;

%% Creation of Initial Empires
InitialCountries = GenerateNewCountry(AlgorithmParams.NumOfCountries , ProblemParams);

% Calculates the cost of each country. The less the cost is, the more is the power.
if isempty(ProblemParams.CostFuncExtraParams)
    InitialCost = feval(ProblemParams.CostFuncName,InitialCountries);    
else
    InitialCost = feval(ProblemParams.CostFuncName,InitialCountries,ProblemParams.CostFuncExtraParams);
end
[InitialCost,SortInd] = sort(InitialCost);                          % Sort the cost in assending order. The best countries will be in higher places
InitialCountries = InitialCountries(SortInd,:);                     % Sort the population with respect to their cost.

Empires = CreateInitialEmpires(InitialCountries,InitialCost,AlgorithmParams, ProblemParams);

%% Main Loop
MinimumCost = repmat(nan,AlgorithmParams.NumOfDecades,1);
MeanCost = repmat(nan,AlgorithmParams.NumOfDecades,1);

if DisplayParams.PlotCost
    axes(DisplayParams.CostAxisHandle);
    if any(findall(0)==DisplayParams.CostFigureHandle)
        h_MinCostPlot=plot(MinimumCost,'r','LineWidth',1.5,'YDataSource','MinimumCost');
        hold on;
        h_MeanCostPlot=plot(MeanCost,'k:','LineWidth',1.5,'YDataSource','MeanCost');
        hold off;
        pause(0.05);
    end
end

for Decade = 1:AlgorithmParams.NumOfDecades
    AlgorithmParams.RevolutionRate = AlgorithmParams.DampRatio * AlgorithmParams.RevolutionRate;

    Remained = AlgorithmParams.NumOfDecades - Decade
    for ii = 1:numel(Empires)
        %% Assimilation;  Movement of Colonies Toward Imperialists (Assimilation Policy)
        Empires(ii) = AssimilateColonies(Empires(ii),AlgorithmParams,ProblemParams);

        %% Revolution;  A Sudden Change in the Socio-Political Characteristics
        Empires(ii) = RevolveColonies(Empires(ii),AlgorithmParams,ProblemParams);
        
        %% New Cost Evaluation
        if isempty(ProblemParams.CostFuncExtraParams)
            Empires(ii).ColoniesCost = feval(ProblemParams.CostFuncName,Empires(ii).ColoniesPosition);
        else
            Empires(ii).ColoniesCost = feval(ProblemParams.CostFuncName,Empires(ii).ColoniesPosition,ProblemParams.CostFuncExtraParams);
        end

        %% Empire Possession  (****** Power Possession, Empire Possession)
        Empires(ii) = PossesEmpire(Empires(ii));
        
        %% Computation of Total Cost for Empires
        Empires(ii).TotalCost = Empires(ii).ImperialistCost + AlgorithmParams.Zeta * mean(Empires(ii).ColoniesCost);
    
    end
    
    %% Uniting Similiar Empires
    Empires = UniteSimilarEmpires(Empires,AlgorithmParams,ProblemParams);

    %% Imperialistic Competition
    Empires = ImperialisticCompetition(Empires);
    
    if numel(Empires) == 1 && AlgorithmParams.StopIfJustOneEmpire
        break
    end

    %% Displaying the Results
    DisplayEmpires(Empires,AlgorithmParams,ProblemParams,DisplayParams);
    
    ImerialistCosts = [Empires.ImperialistCost];
    MinimumCost(Decade) = min(ImerialistCosts);
    MeanCost(Decade) = mean(ImerialistCosts);

    if DisplayParams.PlotCost
        refreshdata(h_MinCostPlot);
        refreshdata(h_MeanCostPlot);
        drawnow;
        pause(0.01);
    end
    
end % End of Algorithm
MinimumCost(end)