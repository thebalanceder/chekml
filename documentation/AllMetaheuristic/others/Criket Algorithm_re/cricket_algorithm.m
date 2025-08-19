
function [best,fmin,N_iter]=cricket_algorithm(para)
% Default parameters
if nargin<1,  para=[25  0.5];  end
n=para(1);  % Population size, typically 10 to 25
alpha=para(2);
betamin=0.2;
d=2;
pi=3.14;
Lb=-5*ones(1,d);
% Upper limit/bounds/ a vector
Ub=10*ones(1,d);
    %Lower Bond


Qmin=0;         % Frequency minimum
      
% Iteration parameters
tol=10^(-6);    % Stop tolerance
N_iter=0;       % Total number of function evaluations
% Dimension of the search variables


% Initial arrays
Q=zeros(n,1);  
v=zeros(n,d);   

for i=1:n,

 Sol(i,:)=Lb+(Ub-Lb).*rand(1,d);
  Fitness(i)=Fun(Sol(i,:));
  
end

% Find the current best
[fmin,I]=min(Fitness);
best=Sol(I,:);
tic;

% Start the iterations -- Cricket Algorithm
while(fmin>tol)
for i=1:n,
            N(i,:) = randi([0 120],1,d);         
           T(i,:)=0.891797*N(i,:)+40.0252;
           if(T(i,:)>180)
               T(i,:)=180;               
           end
           if(T(i,:)<55)
               T(i,:)=55;
           end
           C(i,:)=(5/9)*(T(i,:)-32);           
           V(i,:)=20.1*sqrt(273+C(i,:));  
           V(i,:)=sqrt(V(i,:))/1000;
           Z(i,:)=(Sol(i,:)-best);             
            if(Z(i,:)==0)
               F(i,:)=0;   
           else
              F(i,:)=V(i,:)/Z(i,:); 
               end
           Q(i,:)=Qmin+(F(i,:)-Qmin)*rand;      
          v(i,:)=v(i,:)+(Sol(i,:)-best)*Q(i)+V(i);
          S(i,:)=Sol(i,:)+v(i,:);          
          SumF=sum(F(i,:))/i+10000;
          SumT=sum(C(i,:))/i;
          gamma=CoefCalculate(SumF,SumT);      
                                
           Solo=Sol;             
           scale=(Ub-Lb);       
           for j=1:n,
               if(Fitness(i)<Fitness(j))
                distance=sqrt(sum((Sol(i,:)-Solo(j,:)).^2));
                PS=Fitness(i)*(4*pi*(distance^2));
                Lp=PS+10*log10(1/4*pi*(distance^2));
                Aatm = (7.4 * ( ((F(i,:)^2)*distance)/ (50*(10^(-8)))));
                RLP=Lp-Aatm;
                K=(RLP)*exp(-gamma*distance.^2);
                beta=K+betamin;
                tmpf=alpha.*(rand(1,d)-0.5).*scale; 
               M(i,:)=Sol(i,:).*(1-beta)+Solo(j,:).*beta+tmpf;
                        
          

          else
             M(i,:)=best+0.01*randn(1,d);   
               end  
          end
          
     % Evaluate new solutions
          if(rand >gamma)
          
              u1(i,:)=S(i,:);
          else
             u1(i,:)=M(i,:);
          end
            u1(i,:)=simplebounds(u1(i,:),Lb,Ub);   
          
           
             Fnew=Fun(u1(i,:));
             
    
           if (Fnew<=Fitness(i)) 
                Sol(i,:)=u1(i,:);
                Fitness(i)=Fnew;
          
          % Update the current best
                                
             if Fnew<=fmin,
                best=u1(i,:);
                fmin=Fnew;

             end
                
           end
           
          
                    alpha=alpha_new(alpha);
        end
                N_iter=N_iter+n;
               
end    
                
    
% Output/display
disp(['Number of evaluations: ',num2str(N_iter)]);
disp(['Best =',num2str(min(best))]);
disp(['fmin: ',num2str(fmin)]);
 
toc;
function [Fonksiyon2]=Fun(u)

  [Fonksiyon2]=Sphere(u);
 
function z= Sphere(u)
n = 2;
s = 0;
for j = 1:n
    s = s+u(j)^2; 
end
z = s;


function alpha=alpha_new(alpha)
delta=0.97;
alpha=delta*alpha;


function s=simplebounds(s,Lb,Ub)
  % Apply the lower bound vector
  ns_tmp=s;
  I=ns_tmp<Lb;
  ns_tmp(I)=Lb(I);
  
  % Apply the upper bound vector 
  J=ns_tmp>Ub;
  ns_tmp(J)=Ub(J);
  % Update this new move 
  s=ns_tmp;
