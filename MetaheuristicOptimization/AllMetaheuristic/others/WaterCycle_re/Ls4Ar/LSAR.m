function [red,time] = LSAR(C,D,T)
%LSAR: Local search algorithm for attribute reduction
%Input: A decision table: C,D and the maximum number of iterations T
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
% T is a non-zero integer. We set T=2000 in our paper.
%Output: The attribute reduction: red
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  LSAR source codes version 1.00                                   %
%                                                                   %
%  Developed in MATLAB R2018(a)                                     %
%                                                                   %
%  Author and programmer: Xiaojun Xie                               %
% Homepage: https://www.researchgate.net/profile/Xiaojun_Xie4       %
%                                                                   %
%         e-Mail: xiex34@mcmaster.ca                                %
%                 xiexj@nuaa.edu.cn                                 %
%    This algorithm can only handle categorical and integer data.   %                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stLSAR=cputime;  
[~,numatt]=size(C);
[red] = fastRed(C,D);
disp(['The size of initialization reduct: ',num2str(length(red))]);
pos=PositiveRegion( C,D );
t=0;
while t<T
    a= RandomSelectAtt(red);
    if length(PositiveRegion( C(:,setdiff(red,a)),D ))==length(pos)
        red=setdiff(red,a);
    else
        u=RandomSelectAtt(setdiff(red,a));
        v=RandomSelectAtt(setdiff(1:numatt,setdiff(red,a)));
        if length(PositiveRegion( C(:,union(setdiff(setdiff(red,a),u),v)),D ))==length(pos)
            red=union(setdiff(setdiff(red,a),u),v);
        end
    end 
    t=t+1;
    disp(['Number of evaluations: ',num2str(t)]);
    disp(['The size of reduct: ',num2str(length(red))]);
end
time=cputime-stLSAR;
end

