%{
%% Please refer to the main paper: (Published: 18 August 2022) 
  TITLE : PVS: a new population-based vortex search algorithm with boosted exploration capability using polynomial mutation
 AUTHOR : Tahir SAG (Department of Computer Engineering, Selcuk University, Turkey)
JOURNAL : Neural Computing and Applications, Volume:34, Pages:18211–18287 (2022). 
    DOI : https://doi.org/10.1007/s00521-022-07671-x
          https://link.springer.com/article/10.1007/s00521-022-07671-x
  email : tahirsag@selcuk.edu.tr
%}

%% PVS: Population-based Vortex Search Algorithm
function [Mu, gmin, max_FEs, iter_results] = PVS(psize, dim, FE_max, objfun, err, lowerlimit, upperlimit, opt_f)
    ProbMut=1/dim;
    ProbCross = 1/dim;
    PC=ProbCross;

    UB = upperlimit * ones(1,dim);
    LB = lowerlimit * ones(1,dim);
    
    Mu = 0.5 * (UB + LB); %initial center of the circle
    gmin = inf; %fitness of the global min
    x = 0.1; % x = 0.1 for gammaincinv(x,a) function
    a=1;
    ginv = (1/x)*gammaincinv(x,a); % initially a = 1
    r = ginv * ((upperlimit - lowerlimit) / 2); % initial radius of the circle

    FEs = 0;
    max_FEs = 0;    
    count = 1; %itr counter
    Cs=zeros(psize,dim);  
    vsize = psize/2;

    while FEs < FE_max
 
        %% FISRT PHASE        
        if count==1
%             Cs(1,:) = Mu;
%             C = r.*randn(psize-1,dim);
%             Cs(2:psize,:) = bsxfun(@plus, C, Mu); %candidate solutions

            C = r.*randn(psize,dim);
            Cs = bsxfun(@plus, C, Mu); %candidate solutions
        else
            C = r.*randn(vsize,dim);
            Cs(1:vsize, :) = bsxfun(@plus, C, Mu); %candidate solutions
        end

        %limit the variables
        rand1 = rand(psize,dim)*(upperlimit - lowerlimit) + lowerlimit;
        Cs(Cs < lowerlimit) =  rand1(Cs < lowerlimit);
        
        rand2 = rand(psize,dim)*(upperlimit - lowerlimit) + lowerlimit;
        Cs(Cs > upperlimit) =  rand2(Cs > upperlimit);

        % Evaluate the candidate solutions
        if count==1
            ObjVal = feval(objfun,Cs);
            FEs = FEs + psize;            
        else
            ObjVal(1:vsize) = feval(objfun,Cs(1:vsize,:));
            FEs = FEs + vsize;
        end

        %% Update center     
        fmin = min(ObjVal); % minimum fitness value
        MinFitInd = find(ObjVal == fmin); % find the min. fitness index

        if numel(MinFitInd) > 1
            MinFitInd = MinFitInd(1); % if more than one solution keep one of them
        end

        itrBest = Cs(MinFitInd,1:dim); %iteration best

        if fmin < gmin
            gmin = fmin; % fitness of the best solution found so far
            Mu = itrBest; %best solution found so far
        end
        
        %% cumulative probabilities for roulette wheel selection            
        prob = 0.9 * (max(ObjVal)-ObjVal) + 0.1;
        prob = prob ./ sum(prob);
        prob = cumsum(prob);
        prob = prob ./ prob(end);
          
        %% SECOND PHASE       
        for i = vsize+1:psize
            
            neigbour=length(find(prob<rand))+1;    
            while(i==neigbour)
                neigbour=length(find(prob<rand))+1;
            end 
            
            sol = Cs(i,:);
            param2change=randi(dim);
            for d=1:dim
               if rand < PC || d==param2change
                   sol(d)=Cs(i,d)+(Cs(i,d)-Cs(neigbour,d))*(rand-0.5)*2;
               end
            end

            %  if generated parameter value is out of boundaries, it is shifted onto the boundaries
            ind=find(sol<LB);
            sol(ind)=LB(ind);
            ind=find(sol>UB);
            sol(ind)=UB(ind);
            
            % evaluate new solution
            ObjValSol=feval(objfun,sol);
            FEs = FEs + 1;

            % a greedy selection is applied between the current solution i and its mutant
            if (ObjValSol<ObjVal(i)) 
                Cs(i,:)=sol;
                ObjVal(i)=ObjValSol;
            else    
                [sol, state] = PolyMutation(Cs(i,:), dim, LB, UB, ProbMut);
                if(state>0)
                    ObjValSol = feval(objfun, Cs(i,:));
                    if (ObjValSol<ObjVal(i))
                        Cs(i,:)=sol;
                        ObjVal(i)=ObjValSol;
                    end
                    FEs = FEs + 1; 
                end
            end

        end
        
        %% Update center     
        fmin = min(ObjVal); % minimum fitness value
        MinFitInd = find(ObjVal == fmin); % find the min. fitness index

        if numel(MinFitInd) > 1
            MinFitInd = MinFitInd(1); % if more than one solution keep one of them
        end

        itrBest = Cs(MinFitInd,1:dim); %iteration best

        if fmin < gmin
            gmin = fmin; % fitness of the best solution found so far
            Mu = itrBest; %best solution found so far
        end
        
        % radius decrement process     
        a = (FE_max-FEs) / FE_max; a(a<=0)=0.1;
        ginv = (1/x)*gammaincinv(x,a); % compute the new ginv value
        r = ginv * ((upperlimit - lowerlimit) / 2); %decrease the radius
        
        %% keep results for each iteration
        iter_results(count).gbestF = gmin;
        iter_results(count).gbestX = Mu;      
        count = count + 1; %itr counter
        
        %% Check the termination condition
        if (abs(opt_f - gmin) <= err)
            break;
        end
        
%         fprintf('Iter=%d ObjVal=%g\n',count, gmin);
    end

    if max_FEs == 0
        max_FEs = FEs;
    end
   
end