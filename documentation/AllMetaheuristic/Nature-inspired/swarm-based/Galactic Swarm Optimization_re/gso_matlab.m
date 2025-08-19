clear all;
close all;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Galactic Swarm Optimization - Copy protected by Venkataraman
%Muthiah-Nakarajan and Mathew Mithra Noel
%The code is provided on an 'as is' basis without any form of warranties - stated or implied.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Trials=50;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%--Setting of Parameters and function begin--%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--D--%%----EPmax-----%%----L1-------%%-----L2------%%---N---%%---M--%%
dim=10;epochnumber=5;Iteration1=198;Iteration2=1000;PopSize=5;subpop=10;
%dim=30;epochnumber=5;Iteration1=280;Iteration2=1500;PopSize=5;subpop=20;
%dim=50;epochnumber=9;Iteration1=250;Iteration2=1500;PopSize=5;subpop=20;

f=@rosenbrock;funct='Rosenbrock';xmin=-30;xmax=30;  %cost function



vmin=xmin;
vmax=xmax;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%--Setting of Parameters and function end--%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for sum=1:Trials


    c1 = 2.05*rand;
    c2 = 2.05*rand;
    c3 = 2.05*rand;
    c4 = 2.05*rand;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initialization of position and velocity vectors begins
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 for num=1:subpop
      for p=1:PopSize
          gal(num).particle(p).x = xmin + (xmax-xmin)*rand(dim,1);
          gal(num).particle(p).cost = f(gal(num).particle(p).x);
          gal(num).particle(p).v =  vmin + (vmax-vmin)*rand(dim,1);
          gal(num).particle(p).pbest=xmin + (xmax-xmin)*rand(dim,1);
          if f(gal(num).particle(p).x)<f(gal(num).particle(p).pbest)
              gal(num).particle(p).pbest=gal(num).particle(p).x;
          end
          gal(num).particle(p).pbest_c=f(gal(num).particle(p).pbest);         
      end
      gal(num).xgbest=gal(num).particle(1).pbest;
      gal(num).cgbest=f(gal(num).xgbest);
      for p=2:PopSize
          if f(gal(num).particle(p).pbest)<gal(num).cgbest
              gal(num).xgbest=gal(num).particle(p).pbest;
              gal(num).cgbest=f(gal(num).particle(p).pbest);
          end
      end
 end   
 galaxy_x=gal(1).xgbest;
 galaxy_c=f(galaxy_x);
 for num=2:subpop
     if f(gal(num).xgbest)<galaxy_c
         galaxy_x=gal(num).xgbest;
         galaxy_c=f(gal(num).xgbest);
     end
 end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initialization of position and velocity vector ends
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

count_f=0;        %Initialization of counter for function calls


for epoch=1:epochnumber
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%-----------Level 1-------------%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for num=1:subpop 
       
        for r=0:Iteration1
            
            for p=1:PopSize
                
                r1 = -1+2*rand;
                r2 = -1+2*rand;

                v = c1*r1*(gal(num).particle(p).pbest - gal(num).particle(p).x) + c2*r2*(gal(num).xgbest - gal(num).particle(p).x);

                gal(num).particle(p).v = (1-r/(Iteration1+1))*gal(num).particle(p).v + v;

                gal(num).particle(p).v = max(gal(num).particle(p).v,vmin);

                gal(num).particle(p).v = min(gal(num).particle(p).v,vmax);

                gal(num).particle(p).x = gal(num).particle(p).x +gal(num).particle(p).v;

                gal(num).particle(p).x  = max(gal(num).particle(p).x ,xmin);

                gal(num).particle(p).x  = min(gal(num).particle(p).x ,xmax);

                gal(num).particle(p).cost = f(gal(num).particle(p).x);
                count_f=count_f+1;

                if gal(num).particle(p).cost < gal(num).particle(p).pbest_c
                    gal(num).particle(p).pbest = gal(num).particle(p).x;
                    gal(num).particle(p).pbest_c = gal(num).particle(p).cost;

                    if gal(num).particle(p).pbest_c < gal(num).cgbest
                        gal(num).xgbest = gal(num).particle(p).pbest;
                        gal(num).cgbest = gal(num).particle(p).pbest_c;

                        if gal(num).cgbest < galaxy_c
                            galaxy_x = gal(num).xgbest;         
                            galaxy_c = gal(num).cgbest;
                        end
                    end
                end

            end
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%------Level 2---------------%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for p=1:subpop
        particle(p).x = gal(p).xgbest;
        particle(p).cost = gal(p).cgbest;
        particle(p).pbest=particle(p).x;
        particle(p).pbest_c=gal(p).cgbest;
        particle(p).v=particle(p).x;
    end
    

        for r=0:Iteration2
            for p=1:subpop
    
                r3 = -1+2*rand;
                r4 = -1+2*rand;

                v =  c3*r3*(particle(p).pbest - particle(p).x)+ c4*r4*(galaxy_x - particle(p).x);

                particle(p).v = (1-r/(Iteration2+1))*particle(p).v + v;
         
                particle(p).v = max(particle(p).v,vmin);
        
                particle(p).v = min(particle(p).v,vmax);

                particle(p).x = particle(p).x +particle(p).v;

                particle(p).x = max(particle(p).x,xmin);

                particle(p).x = min(particle(p).x,xmax);
            
                particle(p).cost = f(particle(p).x);
                count_f=count_f+1;
    
                if particle(p).cost < particle(p).pbest_c                
                    particle(p).pbest = particle(p).x;
                    particle(p).pbest_c = particle(p).cost;
                                  
                        if particle(p).pbest_c < galaxy_c     
                            galaxy_x = particle(p).pbest;        
                            galaxy_c = particle(p).pbest_c;
                        end
                end
            end            
        end
fprintf('\n Epoch=%d objfun_val=%e',epoch, galaxy_c);         
end

fprintf('\n f=%s Trial=%d objfun_val=%e',funct,sum,galaxy_c);

x(sum)=galaxy_c;


end

obj_mean=mean(x);
obj_std=std(x);
obj_var=var(x);
fprintf('\n obj_mean=%e \n obj_std=%e \n obj_var=%e \n best_val=%e \n worst_val=%e \n', obj_mean, obj_std, obj_var, min(x),max(x));
fprintf('median=%e \n',median(x));
fprintf('mode=%e \n',mode(x));
fprintf('function_calls=%d \n func_type=%s \n',count_f,funct);

