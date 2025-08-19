


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

%------------------------- Description -----------------------------------%

Partial Reinforcement Optimizer (PRO) is a novel evolutionary optimization 
algorithm. The major idea of the PRO comes from the partial reinforcement 
effect (PRE) theory, which is an evolutionary learning/training theory in 
psychology tells that a learner is intermittently reinforced to learn or
strengthen a specific behavior during the learning/training process. Acc-
ording to this theory, reinforcement patterns significantly impact the r-
esponse rate and strength of the learner during a reinforcement schedule
that states which instances of behavior must be reinforced. In the PRO a-
lgorithm, the PRE theory is mathematically modeled to an evolutionary op-
timization algorithm to solve global optimization problems.

%-------------------------------------------------------------------------%

%----------------------------- Contents ----------------------------------%

The main directory "PRO Matlab Code CEC2017" contains following Contents:

-  PRO_v1.m          ---------------------------------- Matlab code
-  Run PRO.m         ---------------------------------- Matlab code
-  input_data        ---------------------------------- Bencmarcks (Data)
-  cec17_func.cpp    ---------------------------------- Bencmarcks (Data)
-  cec17_func.mexw64 ---------------------------------- Bencmarcks (Data)
-  Readme.txt        ---------------------------------- Text file  (Docs)

%------------------------ Bencmarcks (Data) ------------------------------%

To investigate the performance of PRO algorithm, we used CEC-BC-2017 [1]. 
This test suit is one of the challenging real-parameter numerical optimiz-
ation benchmarks and contains 30 functions including Unimodal, Multimodal, 
Hybrid, and Composition functions.



%-------------------------------------------------------------------------%
%-------------------------  Requiremets   --------------------------------%

You need to have Matlab software version R2013b or higher installed.


%-------------------------------------------------------------------------%
%-------------------------   References   --------------------------------%


[1] Awad, N. H., Ali, M. Z., & Suganthan, P. N. (2017). Ensemble sinusoidal 
    differential covariance matrix adaptation with Euclidean neighborhood 
    for solving CEC2017 benchmark problems. In 2017 IEEE Congress on Evolu-
    tionary Computation (CEC) (pp. 372–379). IEEE. 
    doi:10.1109/CEC.2017.7969336.
