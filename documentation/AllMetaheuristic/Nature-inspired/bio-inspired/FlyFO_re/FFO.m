% Project Title: Flying Foxes Optimization (FFO) in MATLAB
%
% https://link.springer.com/article/10.1007%2Fs00366-021-01554-w
% https://www.researchgate.net/publication/357536823_A_global_optimizer_inspired_from_the_survival_strategies_of_flying_foxes
%
% Developers: K. Zervoudakis & S. Tsafarakis
%
% Contact Info: kzervoudakis@isc.tuc.gr
%               School of Production Engineering and Management,
%               Technical University of Crete, Chania, Greece
%               https://scholar.google.com/citations?user=QZ6WGDoAAAAJ&hl=en
%               https://www.researchgate.net/profile/Konstantinos-Zervoudakis
%
% Researchers are allowed to use this code in their research projects, as
% long as they cite as:
% Zervoudakis, K., Tsafarakis, S. A global optimizer inspired from the 
% survival strategies of flying foxes. Engineering with Computers (2022).
% https://doi.org/10.1007/s00366-021-01554-w
%%
clc; clear; close all;
%% Problem Definition
ObjectiveFunction=@(x) Sphere(x); funcname='Sphere function';
LowerBound=-10; UpperBound= 10; % Decision Variables Lower & Upper Bounds
ProblemSize=[1 50];         % Decision Variables Size
%% Flying Fox Optimization Algorithm Settings
methname='Flying Fox Optimization';
nPop=round(10+2*sqrt(prod(ProblemSize))); % Number of Flying Foxes (Population Size)
deltasO=[0.2 0.4 0.6];
deltasOmax=deltasO;
deltasOmin=[0.02 0.04 0.06];
parameter.alpha=[1 1.5 1.9];
parameter.pa=[0.5 0.85 0.99];
funcevs=100000;
funccount=0; % Function evaluations counter
SurvList=round(nPop/4);
%% Initialization
EmptyIndividuals.Position=[];
EmptyIndividuals.Past.Cost=[];
EmptyIndividuals.Cost=[];
FlyingFox=repmat(EmptyIndividuals,nPop,1); % Initialize Population Array
BestSol.Cost=inf;
for i=1:nPop % Create Initial Flying Foxes
    FlyingFox(i).Position=unifrnd(LowerBound,UpperBound,ProblemSize);
    FlyingFox(i).Cost=ObjectiveFunction(FlyingFox(i).Position);
    FlyingFox(i).Past.Cost=FlyingFox(i).Cost;
    funccount=funccount+1;
    if FlyingFox(i).Cost<=BestSol.Cost
        BestSol=FlyingFox(i);
    end
end
BestSolutions=zeros(round(funcevs/nPop),1); % Array to Hold Best Cost Values
WorstCost=max([FlyingFox.Cost]);
%% Flying Fox Optimization Algorithm Main Loop
it=0;
SurvivalList=FlyingFox;
while funccount<=funcevs
    it=it+1;
    for i=1:nPop
        deltamax=abs(BestSol.Cost-WorstCost);
        deltas=deltasO*deltamax;
        [alpha,pa]=FuzzySelfTuning(BestSol,FlyingFox(i),WorstCost,deltasO,parameter);
        if norm(FlyingFox(i).Cost-BestSol.Cost)>(deltas(1)*0.5)
            z=FlyingFox(i).Position+alpha.*unifrnd(0,1,ProblemSize).*(BestSol.Position-FlyingFox(i).Position);
        else
            A=randperm(nPop); A(A==i)=[]; a=A(1); b=A(2);
            stepsize=unifrnd(0,1,ProblemSize).*(BestSol.Position-FlyingFox(i).Position)+unifrnd(0,1,ProblemSize).*(FlyingFox(a).Position-FlyingFox(b).Position);
            z=zeros(size(FlyingFox(a).Position));
            j0=randi([1 numel(FlyingFox(a).Position)]);
            for j=1:numel(FlyingFox(a).Position)
                if j==j0 || rand>=pa
                    z(j)=FlyingFox(i).Position(j)+stepsize(j);
                else
                    z(j)=FlyingFox(i).Position(j);
                end
            end
        end
        temp=repmat(EmptyIndividuals,1,1);
        FlyingFox(i).Past.Cost=FlyingFox(i).Cost; % Evaluate new FlyingFoxes
        temp.Past.Cost=FlyingFox(i).Past.Cost;
        z=max(z,LowerBound); z=min(z,UpperBound);% Position Limits
        temp.Cost=ObjectiveFunction(z);
        temp.Position=(z);
        funccount=funccount+1;
        if temp.Cost<FlyingFox(i).Cost
            FlyingFox(i)=temp;
            if FlyingFox(i).Cost<=BestSol.Cost
                BestSol=FlyingFox(i);
            end
        end
        if temp.Cost>WorstCost
            WorstCost=temp.Cost;
        end
        SurvivalList=UpdateSurvivalList(SurvivalList,SurvList,temp);
        deltamax=abs(BestSol.Cost-WorstCost);
        deltas=deltasO*deltamax;
        if norm(temp.Cost-BestSol.Cost)>(deltas(3))
            FlyingFox(i)=ReplaceWithSurvivalList(SurvivalList,EmptyIndividuals,ProblemSize);
            FlyingFox(i).Position=max(FlyingFox(i).Position,LowerBound); FlyingFox(i).Position=min(FlyingFox(i).Position,UpperBound);% Position Limits
            FlyingFox(i).Cost=ObjectiveFunction(FlyingFox(i).Position);
            FlyingFox(i).Past.Cost=FlyingFox(i).Cost;
            if FlyingFox(i).Cost<BestSol.Cost
                BestSol=FlyingFox(i);
            end
            if FlyingFox(i).Cost>WorstCost
                WorstCost=FlyingFox(i).Cost;
            end
            funccount=funccount+1;
        end
    end
    %% Suffocating Flying Foxes
    pBestFF=(find([FlyingFox.Cost]==BestSol.Cost==1));
    nBestFF=size(pBestFF,2);
    pDeath=(nBestFF-1)/nPop;
    for i=1:2:nBestFF
        if rand<pDeath
            j = 1:nPop;
            j(pBestFF) = [];
            if mod(nBestFF,2)==1 && i==nBestFF
                FlyingFox(pBestFF(i))=ReplaceWithSurvivalList(SurvivalList,EmptyIndividuals,ProblemSize);
                FlyingFox(pBestFF(i)).Position=max(FlyingFox(pBestFF(i)).Position,LowerBound); FlyingFox(pBestFF(i)).Position=min(FlyingFox(pBestFF(i)).Position,UpperBound);% Position Limits
                FlyingFox(pBestFF(i)).Cost=ObjectiveFunction(FlyingFox(pBestFF(i)).Position);
                FlyingFox(pBestFF(i)).Past.Cost=FlyingFox(pBestFF(i)).Cost;
                SurvivalList=UpdateSurvivalList(SurvivalList,SurvList,FlyingFox(pBestFF(i)));
                if FlyingFox(pBestFF(i)).Cost<BestSol.Cost
                    BestSol=FlyingFox(pBestFF(i));
                end
                if FlyingFox(pBestFF(i)).Cost>WorstCost
                    WorstCost=FlyingFox(pBestFF(i)).Cost;
                end
                funccount=funccount+1;
            else
                parent1=randi(nPop);
                parent2=randi(nPop);
                if rand<0.5 && FlyingFox(parent1).Cost~=FlyingFox(parent2).Cost
                    [FlyingFox(pBestFF(i)).Position, FlyingFox(pBestFF(i+1)).Position]=Crossover(FlyingFox(parent1).Position,FlyingFox(parent2).Position,LowerBound,UpperBound);
                else
                    FlyingFox(pBestFF(i))=ReplaceWithSurvivalList(SurvivalList,EmptyIndividuals,ProblemSize);
                    FlyingFox(pBestFF(i+1))=ReplaceWithSurvivalList(SurvivalList,EmptyIndividuals,ProblemSize);
                end
                FlyingFox(pBestFF(i)).Position=max(FlyingFox(pBestFF(i)).Position,LowerBound); FlyingFox(pBestFF(i)).Position=min(FlyingFox(pBestFF(i)).Position,UpperBound);% Position Limits
                FlyingFox(pBestFF(i)).Cost=ObjectiveFunction(FlyingFox(pBestFF(i)).Position);% Evaluate Solution 1
                FlyingFox(pBestFF(i)).Past.Cost=FlyingFox(pBestFF(i)).Cost;
                SurvivalList=UpdateSurvivalList(SurvivalList,SurvList,FlyingFox(pBestFF(i)));
                if FlyingFox(pBestFF(i)).Cost<BestSol.Cost
                    BestSol=FlyingFox(pBestFF(i));
                end
                if FlyingFox(pBestFF(i)).Cost>WorstCost
                    WorstCost=FlyingFox(pBestFF(i)).Cost;
                end
                funccount=funccount+1;
                FlyingFox(pBestFF(i+1)).Position=max(FlyingFox(pBestFF(i+1)).Position,LowerBound); FlyingFox(pBestFF(i+1)).Position=min(FlyingFox(pBestFF(i+1)).Position,UpperBound);% Position Limits
                FlyingFox(pBestFF(i+1)).Cost=ObjectiveFunction(FlyingFox(pBestFF(i+1)).Position);% Evaluate Solution 2
                FlyingFox(pBestFF(i+1)).Past.Cost=FlyingFox(pBestFF(i+1)).Cost;
                SurvivalList=UpdateSurvivalList(SurvivalList,SurvList,FlyingFox(pBestFF(i+1)));
                if FlyingFox(pBestFF(i+1)).Cost<BestSol.Cost
                    BestSol=FlyingFox(pBestFF(i+1));
                end
                if FlyingFox(pBestFF(i+1)).Cost>WorstCost
                    WorstCost=FlyingFox(pBestFF(i+1)).Cost;
                end
                funccount=funccount+1;
            end
        end
    end
    BestSolutions(it)=BestSol.Cost;
    % Show Iteration Information
    disp([methname ': ' funcname,': Function Evaluations: ' num2str(funccount)  ', Iterations: ' num2str(it) ', Best Cost = ' num2str(BestSol.Cost)]);
    deltasO=deltasOmax-((deltasOmax-deltasOmin)./funcevs).*funccount;
end
%% Results
figure;
plot(BestSolutions,'LineWidth',2); semilogy(BestSolutions,'LineWidth',2);
xlabel('Iterations'); ylabel(funcname); grid on;
%% Functions
function z=Sphere(x)
z=sum(x.^2);
end
%%
function SurvivalList=UpdateSurvivalList(SurvivalList,SurvList,temp)
if temp.Cost<SurvivalList(end).Cost
    SurvivalList=[temp;SurvivalList];
    [~,ii]=unique((([SurvivalList.Cost])));
    SurvivalList=SurvivalList(ii);
    if size(SurvivalList,1)>SurvList
        SurvivalList=SurvivalList(1:SurvList);
    end
end
end
%%
function ffox=ReplaceWithSurvivalList(SurvivalList,EmptyIndividuals,ProblemSize)
m=randi([2 size(SurvivalList,1)]);
ffox=repmat(EmptyIndividuals,1,1);
h=randperm(size(SurvivalList,1),m);
ffox.Position=zeros(ProblemSize);
for i=1:size(h,2)
    ffox.Position=ffox.Position+SurvivalList(h(i)).Position;
end
ffox.Position=ffox.Position/m;
end
%%
function [off1, off2]=Crossover(x1,x2,LowerBound,UpperBound)
extracros=0.0;
L=unifrnd(-extracros,1+extracros,size(x1));
off1=L.*x1+(1-L).*x2; off2=L.*x2+(1-L).*x1;
off1=max(off1,LowerBound); off1=min(off1,UpperBound);% Position Limits
off2=max(off2,LowerBound); off2=min(off2,UpperBound);
end
%% Fuzzy Self tuning method
function [alpha,pa]=FuzzySelfTuning(BestSol,FlyingFox,WorstCost,deltasO,parameter)
delta=norm(BestSol.Cost-FlyingFox.Cost);
deltamax=abs(BestSol.Cost-WorstCost);
fi=((FlyingFox.Cost-FlyingFox.Past.Cost)/(deltamax));
%% delta membership
deltas=deltasO*deltamax;
parametercount=0; ruleno=zeros(2,3); %(2 is the number of parameters)
membership.delta.Same.Logic=false; membership.delta.Near.Logic=false; membership.delta.Far.Logic=false;
if 0<=delta && delta<deltas(2)
    membership.delta.Far.Logic=false; membership.delta.Far.Value=0;
    if delta<deltas(1)
        membership.delta.Same.Logic=true; membership.delta.Same.Value=1;
        membership.delta.Near.Logic=false; membership.delta.Near.Value=0;
    end
end
if deltas(1)<=delta && delta<=deltamax
    if deltas(1)<=delta && delta<deltas(2)
        membership.delta.Same.Logic=true; membership.delta.Same.Value=(deltas(2)-delta)/(deltas(2)-deltas(1));
        membership.delta.Near.Logic=true; membership.delta.Near.Value=(delta-deltas(1))/(deltas(2)-deltas(1));
    end
    if deltas(2)<=delta && delta<=deltamax
        membership.delta.Same.Logic=false; membership.delta.Same.Value=0;
        if deltas(2)<=delta && delta<=deltas(3)
            membership.delta.Near.Logic=true; membership.delta.Near.Value=(deltas(3)-delta)/(deltas(3)-deltas(2));
            membership.delta.Far.Logic=true; membership.delta.Far.Value=(delta-deltas(2))/(deltas(3)-deltas(2));
        end
        if deltas(3)<=delta && delta<=deltamax
            membership.delta.Near.Logic=false; membership.delta.Near.Value=0;
            membership.delta.Far.Logic=true; membership.delta.Far.Value=1;
        end
    end
end
%% fi membership
membership.fi.Better.Logic=false; membership.fi.Better.Value=0;
membership.fi.Same.Logic=true; membership.fi.Same.Value=1-abs(fi);
membership.fi.Worse.Logic=false; membership.fi.Worse.Value=0;
if fi>=-1 && fi<=1
    if fi>=-1 && fi<0
        if fi==-1
            membership.fi.Better.Logic=true; membership.fi.Better.Value=1;
        elseif fi>-1 && fi<0
            membership.fi.Better.Logic=true; membership.fi.Better.Value=-fi;
        end
    elseif 0<=fi && fi<=1
        if fi==1
            membership.fi.Worse.Logic=true; membership.fi.Worse.Value=1;
        elseif fi<1 && fi>0
            membership.fi.Worse.Logic=true; membership.fi.Worse.Value=fi;
        end
    end
end
%% alpha
parametercount=parametercount+1; rulecount=1;
if membership.fi.Better.Logic==true
    ruleno(parametercount,rulecount)=membership.fi.Better.Value;
else
    ruleno(parametercount,rulecount)=0;
end
rulecount=rulecount+1;
if membership.fi.Same.Logic==true
    ruleno(parametercount,rulecount)=membership.fi.Same.Value;
elseif membership.delta.Same.Logic==true
    ruleno(parametercount,rulecount)=membership.delta.Same.Value;
elseif membership.delta.Near.Logic==true
    ruleno(parametercount,rulecount)=membership.delta.Near.Value;
else
    ruleno(parametercount,rulecount)=0;
end
rulecount=rulecount+1;
if membership.fi.Worse.Logic==true
    ruleno(parametercount,rulecount)=membership.fi.Worse.Value;
elseif membership.delta.Far.Logic==true
    ruleno(parametercount,rulecount)=membership.delta.Far.Value;
else
    ruleno(parametercount,rulecount)=0;
end
alpha=(sum(ruleno(parametercount,:).*parameter.alpha))/(sum(ruleno(parametercount,:)));
%% pa
parametercount=parametercount+1; rulecount=1;
if membership.fi.Worse.Logic==true
    ruleno(parametercount,rulecount)=membership.fi.Worse.Value;
elseif membership.delta.Far.Logic==true
    ruleno(parametercount,rulecount)=membership.delta.Far.Value;
end
rulecount=rulecount+1;
if membership.fi.Same.Logic==true
    ruleno(parametercount,rulecount)=membership.fi.Same.Value;
elseif membership.delta.Same.Logic==true
    ruleno(parametercount,rulecount)=membership.delta.Same.Value;
end
rulecount=rulecount+1;
if membership.fi.Better.Logic==true
    ruleno(parametercount,rulecount)=membership.fi.Better.Value;
elseif membership.delta.Near.Logic==true
    ruleno(parametercount,rulecount)=membership.delta.Near.Value;
end
pa=(sum(ruleno(parametercount,:).*parameter.pa))/(sum(ruleno(parametercount,:)));
end