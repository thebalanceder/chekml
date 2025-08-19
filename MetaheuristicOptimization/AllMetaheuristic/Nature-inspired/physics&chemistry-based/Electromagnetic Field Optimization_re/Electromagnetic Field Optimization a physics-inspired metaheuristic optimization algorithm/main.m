%**************************************************************************************************
%Reference:  Abedinpourshotorban, H., Shamsuddin, S. M., Beheshti, Z., & Jawawi, D. N. (2015). 
%            Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm. 
%            Swarm and Evolutionary Computation..
%
% Note: developed by Abedinpourshotorban, H. (2015)
%**************************************************************************************************

clc;
clear all;
N_var=50;
N_emp=50;
Max_gen=50000;
minval= -100;
maxval= 100;
R_rate=0.3;
Ps_rate=0.2;
P_field=0.1;
N_field=0.45;

runs=30;

for i=1:30
    func_num=i;
    for j=1:runs
        i,j,
        errEFO(i,j) = EFO (N_var,N_emp,Max_gen,minval,maxval,R_rate,Ps_rate,P_field,N_field,func_num);
        errEFO(i,j)
    end
    %abc
    mean_errEFO(i)=mean(errEFO(i,:));
    std_errEFO(i)=std(errEFO(i,:));
    median_errEFO(i)=median(errEFO(i,:));
   
    save EFO_30d_result
end
