
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

clc
clear
close all
Fun_name='F1';                     % number of test functions: 'F1' to 'F23'
N=50;                     % number of pop
Max_Iteration=1000;                     % maximum number of iteration
[lb,ub,dim,func]=fun_info(Fun_name);                     % Object function
[BestCost,BestSol,cg_curve]=DRA(N,Max_Iteration,lb,ub,dim,func);

display(['The best solution obtained by DRA for ' [num2str(Fun_name)],'  is : ', num2str(BestSol)]);
display(['The best optimal value of the objective funciton found by DRA  for ' [num2str(Fun_name)],'  is : ', num2str(BestCost)]);

figure=gcf;
semilogy(cg_curve,'Color','#b28d90','LineWidth',2)
xlabel('Iteration');
ylabel('Best score obtained so far');
box on
set(findall(figure,'-property','FontName'),'FontName','Times New Roman')
legend('DRA')

