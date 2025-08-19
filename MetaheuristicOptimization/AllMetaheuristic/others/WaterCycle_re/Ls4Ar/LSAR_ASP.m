function [red,time] = LSAR_ASP(C,D)
%LSAR-ASP: A local search algorithm with the attribute pair selection
% mechanism for attribute reduction
%Input: A decision table: C,D
% C is the attribute set  and D is the decision attribute
% An example of input:
% C=[1 1 1 1 1 1      
%    1 1 2 1 1 2
%    1 1 1 1 2 2
%    2 1 2 2 2 1
%    1 2 2 3 1 2
%    1 2 3 1 2 2
%    1 3 3 1 2 1]; 
% D=[1;1;1;2;2;2;1];
%Output: The attribute reduction: red
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  LSAR-ASP source codes version 1.00                                   %
%                                                                   %
%  Developed in MATLAB R2018(a)                                     %
%                                                                   %
%  Author and programmer: Xiaojun Xie                               %
% Homepage: https://www.researchgate.net/profile/Xiaojun_Xie4       %
%                                                                   %
%         e-Mail: xiex34@mcmaster.ca                                %
%                 xiexj@nuaa.edu.cn                                 %
%    This algorithm can only handle categorical and integer data.   % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stLSAR=cputime;  
%[~,numatt]=size(C);
RemoveSet=[];
[red] = fastRed(C,D);
disp(['The size of initialization reduct: ',num2str(length(red))]);
pos=PositiveRegion( C,D );
t=0;
while length(red)~=length(RemoveSet)
    a= RandomSelectAtt(setdiff(red,RemoveSet));
    if length(PositiveRegion( C(:,setdiff(red,a)),D ))==length(pos)
        red=setdiff(red,a);
    else
        [u,v]=APSmechanism(C,D,setdiff(red,a));
        if u==0&&v==0
            RemoveSet=union(RemoveSet,a);
        else
            red=union(setdiff(setdiff(red,a),u),v);
             RemoveSet=[];
        end
    end 
    t=t+1;
    disp(['Number of evaluations: ',num2str(t)]);
    disp(['The size of reduct: ',num2str(length(red))]);
end
time=cputime-stLSAR;
end

