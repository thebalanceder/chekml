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

% This function initialize the first population of search agents
function Positions=initialization(SearchAgents_no,dim,ub,lb)

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
%   Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;   
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end