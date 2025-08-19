%_________________________________________________________________________
%  Walrus Optimizer (WO) source code (Developed in MATLAB R2023a)
%  Source codes demo version 1.0
%  programming: Muxuan Han & Qiuyu Yuan
%
%  Please refer to the main paper:
%  Muxuan Han, Zunfeng Du, Kumfai Yuen, Haitao Zhu, Yancang Li, Qiuyu Yuan. 
%  Walrus Optimizer: A novel nature-inspired metaheuristic algorithm, 
%  Expert Systems with Applications, November 2023, 122413. 
%  https://doi.org/10.1016/j.eswa.2023.122413
%  
%  E-mails: hanmuxuan@tju.edu.cn         (Muxuan Han)
%           dzf@tju.edu.cn               (Zunfeng Du)
%           kumfai.yuen@ntu.edu.sg       (Kumfai Yuen)
%           htzhu@tju.edu.cn             (Haitao Zhu) 
%           liyancang@hebeu.edu.cn       (Yancang Li)
%           yuanqiuyu@tju.edu.cn         (Qiuyu Yuan)
%_________________________________________________________________________
clear all
clc

populationSize = 100; 
Max_iteration = 2000;
runs = 100;

for fn = 1

    Function_name=strcat('F',num2str(fn));
    [lb,ub,dim,fobj]=Get_Functions_details(Function_name);
    Best_score_T = zeros(1,runs);
    for run=1:runs
        rng('shuffle');
        [Best_score,Best_pos,PO_cg_curve]=WO(populationSize,Max_iteration,lb,ub,dim,fobj);
        Best_score_T(1,run) = Best_score;
    end

    Best_score_Best = min(Best_score_T);
    Best_score_Worst = max(Best_score_T);
    Best_score_Median = median(Best_score_T,2);
    Best_Score_Mean = mean(Best_score_T,2);
    Best_Score_std = std(Best_score_T);

    display(['Fn = ', num2str(fn)]);
    display(['Best, Worst, Median, Mean, and Std. are as: ', num2str(Best_score_Best),'  ', ...
    num2str(Best_score_Worst),'  ', num2str(Best_score_Median),'  ', num2str(Best_Score_Mean),'  ', num2str(Best_Score_std)]);

end
