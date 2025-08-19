
%% Developer: Hamza YAPICI
% This version of ES-PSO algorithm for optimization can be also used for 
% benchmark functions. 
%Hamza Yapýcý and Nurettin Çetinkaya, 
%“An Improved Particle Swarm Optimization Algorithm Using Eagle Strategy for Power Loss Minimization,” 
%Mathematical Problems in Engineering, vol. 2017, 
%Article ID 1063045, 11 pages, 2017. https://doi.org/10.1155/2017/1063045.
%%1.724852784871021
clear all
clc
format long;
fhd = @Obj_function3; % Objective Function: Obj_function-Welded Beam Design, 
                        % Obj_function2-Tension/Compression, Obj_function3-pressure vessel,
                        % Obj_function4-cantilever beam design
popsize = 300; 
npar = 4; 
maxit = 1000;

% pressure vessel
Lb=[0 0 10 10];
Ub=[99 99 200 200];

% Welded beam
% Lb=[.1 .1 .1 .1];
% Ub=[2 10 10 2];

% cantilever beam
% Lb=[0 0 0 0 0];
% Ub=[100 100 100 100 100];

% Tension/Compression
% Lb=[.05 .25 2];
% Ub=[2 1.3 15];

c1 = 2; % acceleration coefficient
c2 = 2; % acceleration coefficient 
wmax=0.9; % max weight
wmin=0.4; % min weight
for j=1:npar
    par(:,j)=Lb(j)+(Ub(j)-Lb(j)).*rand(popsize,1);

end
% par=Lb+(Ub-Lb).*rand(popsize,npar); % random population
% minpar=min(par);
% maxpar=max(par);
vel = rand(popsize,npar); % random velocities
% velmax = 0.2 * (maxpar - minpar);
% velmin = -velmax;
% Evaluate initial population
for ik=1:popsize
    cost(ik)=fhd(par(ik,:)); % calculates population cost
end
minc(1)=min(cost); % minimum cost
meanc(1)=mean(cost); % mean cost
globalmin=minc(1); % Initial global minimum
% Initial local particle
localpar = par; 
localcost = cost; 
% Best particle
[globalcost,indx] = min(cost);
globalpar=par(indx,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start iterations
iter = 0; 
while iter < maxit
    iter = iter + 1;
    p=.2;
    a=rand;
    if p<=a
        r1 = rand(popsize,npar); % random numbers
        r2 = rand(popsize,npar); % random numbers1
        w=wmax-((wmax-wmin)/maxit)*iter; %inertia weight
        vel=(w*vel + c1 *r1.*(localpar-par) + c2*r2.*(ones(popsize,1)*globalpar-par));    
        par = par + vel; % update of position
        for j=1:npar
                for i=1:popsize           
                    if par(i,j)<Lb(j)
                        par(i,j)=Lb(j);
                    end
                    if par(i,j)>Ub(j)
                        par(i,j)=Ub(j);
                    end
                end
        end
        for ik=1:popsize
            c_x(ik)=fhd(par(ik,:)); % new cost
        end
    %         Update local pars
        for h=1:popsize
            if c_x(h)<localcost(h)
                localcost(h)=c_x(h);
                localpar(h,:)=par(h,:);
            else
                localcost(h)=localcost(h);
                localpar(h,:)=localpar(h,:);
            end
        end
    end
    % by Xin-She Yang and Suash Deb
    % Levy flight (if step=1, then random walk)
    beta=1.5;
    alpha=randn;
    sig=beta.*(gamma(1+beta)*sin(pi*beta/2))/pi;
    for j=1:popsize,
        s(j,:)=localpar(j,:);        
        u=(alpha)*sig;
        st=rand;
        step=u./(st.^(1+beta));     
        aL=0.1*step.*(s(j,:)-globalpar);      
        s(j,:)=s(j,:)+aL;     
    end
    for j=1:npar
                for i=1:popsize
                    
                    if s(i,j)<Lb(j)
                        s(i,j)=Lb(j);
                    end
                    if s(i,j)>Ub(j)
                        s(i,j)=Ub(j);
                    end
                end
    end
    for ik=1:popsize
        new_fit(ik)=fhd(s(ik,:)); % new cost
    end
%   Update global par
    [global_old, i1]=min(localcost);
    [global_new, i2]=min(new_fit);
    globpar1=localpar(i1,:);
    globpar2=s(i2,:);
    if global_new<global_old
        new_cost=global_new;
        new_par=globpar2;
    else
        new_cost=global_old;
        new_par=globpar1;
    end
    if new_cost<globalcost
        globalcost=new_cost;
        globalpar=new_par;
    else 
        globalcost=globalcost;
        globalpar=globalpar;
    end
    [iter globalpar globalcost]; % print output each iteration
    minc(iter+1)=globalcost; % minimum for this iteration
    globalmin(iter+1)=globalcost; 
    meanc(iter+1)=mean(localcost); 
    disp(['Iteration : ' num2str(iter) ...
        '- Best Cost = ' num2str(minc(iter+1))]);
    
end
ans_1=globalcost
ans_2=globalpar;
% meanc=mean(minc)
% toc
figure(3)
iters=0:length(minc)-1;
plot(iters,minc);
xlabel('iteration');ylabel('fit_val');
text(0,minc(1),'best');

