%Future search algorithm for optimization
% By M. Elsisi
%Cite this article
% Elsisi, M. Future search algorithm for optimization. Evol. Intel. 12, 21–31 (2019). https://doi.org/10.1007/s12065-018-0172-2   
clc
clear
n=30;      % Population size
iteration=1000;  % Maximum number of "iterations"
r_time=30; % Number of runtime
d=1000;       % Number of dimensions 
% Lower limit/bounds/ a vector
 Lb=-100*ones(1,d);
% Upper limit/bounds/ a vector
 Ub=100*ones(1,d);

for r=1:r_time; 
% Initialize the population/solutions
for i=1:n,
  S(i,:)=Lb+(Ub-Lb).*rand(1,d);
  for k=1:d
            if S(i,k)>Ub(k), S(i,k)=Ub(k); end
            if S(i,k)<Lb(k), S(i,k)=Lb(k); end
        end
  Fitness(i)=Fun(S(i,:));
end
% Find the initial best solution
[fmin,I]=min(Fitness);
best=S(I,:);
Lbe=S;
Lbest=Fitness;
%%main global loop
iter=0;                      % Iterations’ counter
for t=1:iteration, 
    
    %%main local loop   
    for i=1:n,
             S(i,:)=S(i,:)+(-S(i,:)+best)*rand+(-S(i,:)+Lbe(i,:))*rand;
          for k=1:d
            if S(i,k)>Ub(k), S(i,k)=Ub(k); end
            if S(i,k)<Lb(k), S(i,k)=Lb(k); end
        end
          
     % Evaluate new solutions
           Fnew(i)=Fun(S(i,:));
     % Update the loacal best solution
           if (Fnew(i)<=Lbest(i)) 
                Lbe(i,:)=S(i,:);
                Lbest(i)=Fnew(i);
           end

          % Update the current global best solution
          if Fnew(i)<=fmin,
                best=S(i,:);
                fmin=Fnew(i);
          end
          
        end

        % loop of  the initial update
        for i=1:n,
             Si(i,:)=best+(best-Lbe(i,:)).*rand;
        for k=1:d
            if Si(i,k)>Ub(k), Si(i,k)=Ub(k); end
            if Si(i,k)<Lb(k), Si(i,k)=Lb(k); end
        end
             Fitness(i)=Fun(Si(i,:));
        % Update the loacal best solution
             if ( Fitness(i)<=Fnew(i)) 
                S(i,:)=Si(i,:);
                Lbe(i,:)=Si(i);
             end
       end
        [fmini,I]=min(Fitness);
besti=Si(I,:);
if fmini<=fmin,
                best=besti;
                fmin=fmini;
          end
       iter=iter+1;
       fgbest(iter)=fmin;
       iter_counter(iter)=iter;
end
% Output/display for each runtime
disp(['Number of Iterations: ',num2str(iter)]);
disp(['Best =',num2str(best),' fmin=',num2str(fmin)]);
gbest_r(r,:)=fgbest;
best_r(r,:)=best;
gbest(r)=min(fgbest);
end

[gbestscore,I]=min(gbest);
bestposition=best_r(I,:);
semilogy(gbest_r(I,:),'-r');

disp(['Number of runtime: ',num2str(r)]);
disp(['Best =',num2str(bestposition),' fmin=',num2str(gbestscore)]);
