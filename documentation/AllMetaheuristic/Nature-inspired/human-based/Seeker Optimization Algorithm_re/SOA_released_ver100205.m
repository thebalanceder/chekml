% Human group optimization (originally named as Seeker optimization algorithm) by Chaohua Dai,
%
% Dr. Chaohua Dai
% Associate Professor 
% The School of Electrical Engineering,
% Southwest Jiaotong University,
% Chengdu 610031, China
% Office: Tel: +86-28-87603332, Fax: 
% Email: dchzyf@yahoo.com.cn
% http://www.sciencenet.cn/u/dchzyf/
% 
% (The author has re-named SOA as human group optimizization, please pay attention to out paper for this change)

% 

% -----------------------------------------
%            References:
%[1] Chaohua Dai, Weirong Chen, and Yunfang Zhu. Seeker optimization algorithm for digital IIR filter design, IEEE Transactions on Industrial Electronics, (2009), doi:10.1109/TIE.2009.2031194, in press. Available at: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5229209&isnumber=4387790 
%[2] Chaohua Dai, Weirong Chen, Yunfang Zhu and Xuexia Zhang. Seeker optimization algorithm for optimal reactive power dispatch, IEEE Transactions on Power Systems, 2009, 24(3):1218-1231.
%[3] Chaohua Dai, Weirong Chen, Yunfang Zhu and Xuexia Zhang. Reactive power dispatch considering voltage stability with seeker optimization algorithm, Electric Power System Research, 2009, 79(10), pp.1462-1471. 
%[4] Chaohua Dai, Weirong Chen, Lixiang Li, et al. Seeker optimization algorithm for parameter estimation of time-delay chaotic systems, Physical Review E, 83, 036203(2011)
%[5] Chaohua Dai, Weirong Chen, Lili Ran, Yi Zhang and Yu Du. Human Group Optimizer with Local Search, Lecture Notes in Computer Science, 2011, Volume 6728/2011, 310-320.
%------------------------------------------


% Welcome you to modify the algorithm for its performance improvement.


clear all;
close all;
rand('state',sum(100*clock)) % create a random seed
% clc

%format long


    fname = 'ObjFcn_exa1'; % objective function
    D=3;        
    Size=3*floor(10+2*sqrt(D));
    Variablesmax=[1 30 108];
    Variablesmin=[0.2 22 100];
    initXmax=Variablesmax;
    initXmin=Variablesmin;
    
runs=30;

Max_Gens=300;%round(Max_FES/Size);
Max_FES=Max_Gens*Size;

numregion=3;
for region=1:1:numregion
    startreg(region)=floor((region-1)*Size/numregion+1);
    endreg(region)=floor(region*Size/numregion);
    Sizereg(region)=endreg(region)-startreg(region)+1;
end

rmax=0.5*(Variablesmax-Variablesmin);
rmin=-rmax;

% clear Xmax Xmin initXmax0 initXmin0;

E=zeros(Size,D);
E_Temp = zeros(Size,D);
x_offspring=zeros(1,D);

FuncValueCurve=inf*ones(runs,Max_FES);

gbest_curve_res = [];
gbest_curve = zeros(Max_FES,D);

Time_curve_res = zeros(runs,Max_FES);
Time_curve = zeros(1,Max_FES);

pBestFun=zeros(1,Size);
pBestS=zeros(Size,D); %
gBestS=zeros(1,D);
lBestSreg=zeros(numregion,D); %
lBestFunreg=inf*ones(1,numregion);

Sign_x=ones(1,D);
%Sign_x0=Sign_x;

mu_max=0.99;mu_min=0.0111;
% mu_max=0.95;mu_min=0.1;
w_min=0.4;w_max=0.9;  %

final_gBestFun = zeros(1,runs);
% final_gMinPloss = zeros(1,runs);

final_gBestS = zeros(runs,D);

errorPrev = 0;
FailFES = 0;

pop_temp = zeros(1,D);

for run=1:runs,
    time2=cputime;
    FES=0;
    GENs=0;
    gBestFun=inf;
    startTime = cputime;
       
%     E=ones(Size,1)*initXmin+(ones(Size,1)*(initXmax-initXmin)).*rand(Size,D);
    iBest = 1;
    for s=1:1:Size
%         if s == 1
%             E(s,:) = E0;
%         else
        E(s,:) = Variablesmin + rand(1,D).*(Variablesmax - Variablesmin);
%         end
               
        FES=FES+1;
        % calculate fitness
        tempval = feval(fname,E(s,:)); 
            
        F(s)=tempval;
        

            if F(s)<=gBestFun
                gBestFun=F(s);
                gBestS = E(s,:);
            end   
            
            FuncValueCurve(run,FES)=gBestFun;
            gbest_curve_res(run).gbest_curve(FES,:) = gBestS;
            Time_curve(FES)= cputime - startTime;
            
    end
     
    errorPrev = gBestFun; 

    E_t_1=E; %
    E_t_2=E; %
    
    F_t_1=F; %
    F_t_2=F; %
    
    pBestFun=F;
    pBestS=E;
    
    for region=1:1:numregion,
        [BestFunreg,IndexBestreg]=min(F(startreg(region):endreg(region)));
%         [BestFunreg,IndexBestreg]=min(pBestFun(startreg(region):endreg(region)));
        IndexBestreg=IndexBestreg(1)+startreg(region)-1;
%         if lBestFunreg(region)>=BestFunreg,% 
            lBestFunreg(region)=BestFunreg(1);%
            lBestSreg(region,:)=E(IndexBestreg(1),:); %
%             lBestSreg(region,:)=pBestS(IndexBestreg(1),:); %
%         end
    end


while FES < Max_FES 
    GENs=GENs+1;
    
   weight=w_max-GENs*(w_max-w_min)/Max_Gens;

      
      for region=1:1:numregion,      
          [OderF,IndexF]=sort(F(startreg(region):endreg(region)),'descend');
          IndexF=IndexF+startreg(region)-1;


              rand_En=max(1,ceil(rand.*(Sizereg(region)-1))); 
              En_Temp=weight*abs(E(IndexF(end),:)-E(IndexF(rand_En),:));
% optionalble
%           case 10 % !!! 
%               index_cut = floor(2*(Sizereg(region)-1)/3);
%               rand_En=max(1,ceil(rand*index_cut));
%               En_Temp=weight*abs(E(IndexF(end),:)-E(IndexF(rand_En),:)); 
     
       mu_max=0.95+GENs*(0.99-0.95)/Max_Gens;
      
      for s=startreg(region):1:endreg(region), %  kk+old_num-1,% %开始对种群逐个个体进行进化操作
         
          IndexF_local=find(IndexF==s)+startreg(region)-1;  
          mu=(mu_max-(mu_max-mu_min)*(Size-IndexF_local)/(Size-1));

      mu=mu+(1-mu).*rand(1,D);
  
                x_pdirect=pBestS(s,:)-E(s,:); 

% 2009.07.10                
%                 flag_ldirect=0;
%                 switch flag_ldirect
%                     case 0
                        if lBestFunreg(region)<F(s)
                            g1=1;
                            x_ldirect1=lBestSreg(region,:)-E(s,:); 
                        else
                            g1=0;
                            x_ldirect1=zeros(1,D);
                        end
                        if OderF(end)<F(s)
                            g2=1;
                            x_ldirect2=E(IndexF(end),:)-E(s,:); 
                        else
                            g2=0;
                            x_ldirect2=zeros(1,D);
                        end
          

% 2009.07.10
%             flag_tdirect = 1;
%             if flag_tdirect == 1
                [order_tdirect,index_tdirect]=sort([F_t_2(s) F_t_1(s) F(s)]); %
                E_tdirect=[E_t_2(s,:);E_t_1(s,:);E(s,:)];
                x_tdirect=E_tdirect(index_tdirect(1),:)-E_tdirect(index_tdirect(3),:);
%             else        
%                 if F(s)<=F_t_1(s),           %
%                     x_tdirect=E(s,:)-E_t_1(s,:);
%                 else
%                     x_tdirect=E_t_1(s,:)-E(s,:);
%                 end
%             end


                          p=1;
                          t=1;   
%                           g1 = 0;

%                           flag_directVector=[t p n g1 g2];
                          flag_directVector=[t p g1 g2];
                          
                        tSign_x=sign(x_tdirect);
                        pSign_x=sign(x_pdirect);
                        l1Sign_x=sign(x_ldirect1);
                        l2Sign_x=sign(x_ldirect2);
%                         nSign_x=sign(x_ndirect); 
%                         xSign_x=[tSign_x;pSign_x;nSign_x;l1Sign_x;l2Sign_x];
                        xSign_x=[tSign_x;pSign_x;l1Sign_x;l2Sign_x];
                        
                        index_SelectSign=find(flag_directVector>0);                                                
                        num_Sign = length(index_SelectSign);
                        num_pOne=0;num_nOne=0;
                        for ii=1:num_Sign
                            num_pOne = num_pOne + (abs(xSign_x(index_SelectSign(ii),:))+xSign_x(index_SelectSign(ii),:))/2;
                            num_nOne = num_nOne + (abs(xSign_x(index_SelectSign(ii),:))-xSign_x(index_SelectSign(ii),:))/2;
                        end
                        num_Zeros = num_Sign - (num_pOne + num_nOne);                                                       
                        
                        prob_pOne = num_pOne./num_Sign;
                        prob_nOne = (num_pOne+num_nOne)./num_Sign;
                        prob_Zeros = 1;
                        rand_roulette = rand(1,D);
                        x_direct = (rand_roulette<=prob_pOne).*ones(1,D) + (rand_roulette>prob_pOne & rand_roulette<=prob_nOne).*(-ones(1,D)) ...
                                   + (rand_roulette>prob_nOne & rand_roulette<=prob_Zeros).*zeros(1,D);


                
                Sign_x=sign(x_direct);
                
                index_Ex=find(E(s,:)>Variablesmax);
                Sign_x(index_Ex)=-1;
                index_Ex=find(E(s,:)<Variablesmin);
                Sign_x(index_Ex)=1; 
                index_Sign=find(Sign_x==0);
                %--------------------------------------
                num_Sign=length(index_Sign);
                Sign_x(index_Sign)=round(rand(1,num_Sign)); 
                index_Sign=find(Sign_x==0);
                Sign_x(index_Sign)=-1;
                %--------------------------------------
                %Sign_x(s,index_Sign)=Sign_x0(s,index_Sign);
                
                rTEMP=Sign_x.*(En_Temp.*(-2*log(mu)).^0.5);
                rTEMP=max(min(rmax,rTEMP),rmin);
                
                x_offspring=E(s,:)+rTEMP;          
                x_offspring=max(min(x_offspring,Variablesmax),Variablesmin);               
                
                E_Temp(s,:) = x_offspring;          

                
   
      end % end of "for s=1:1:Size, "
      
      %% 邻域间交流 method two C1 其实，各子群的最佳个体间也可以实行此增强学习Reinforcement Learning策略（此程序没加这个）
      si=0;
      for region1=1:1:numregion,
         if region1~=region,
             si=si+1;
             flag_cross = rand(1,D) < 0.5;
             flag_NOcross = 1-flag_cross;
%              E_Temp(IndexF(si),:) = lBestSreg(region1,:); % rejected !!!
             E_Temp(IndexF(si),:) = flag_NOcross.*E_Temp(IndexF(si),:) + flag_cross.*lBestSreg(region1,:);
         end
      end % region
            
    for s = 1:Size
        x_offspring = E_Temp(s,:);
   
        tempval = feval(fname,x_offspring);    
        
       
                FES=FES+1;
                
                Time_curve(FES)= cputime - startTime;
                
%         if tempval<=F(s)
                    F_t_2(s)=F_t_1(s); 
                    F_t_1(s)=F(s); %上一代函数值
                    F(s)=tempval;
                    E_t_2(s,:)=E_t_1(s,:);
                    E_t_1(s,:)=E(s,:);
                    E(s,:)=x_offspring;
             
                if tempval<=pBestFun(s),
                    pBestFun(s)=tempval;
                    pBestS(s,:)=x_offspring;
                end
%          end

                
                if tempval<=gBestFun
                    gBestFun=tempval;
%                   
                    gBestS=x_offspring;
                    

                end  
                if FES > Max_FES 
                    FuncValueCurve(run,Max_FES)=gBestFun;
                    gbest_curve_res(run).gbest_curve(Max_FES,:) = gBestS;

                else
                    FuncValueCurve(run,FES)=gBestFun;
                    gbest_curve_res(run).gbest_curve(FES,:) = gBestS;
                end
            
 
                
    end
    

    
    %% 领域交流 method three
%     if mod(GENs,5)==0 % C21
%     if errorPrev <= gBestFun % C2
%     if errorPrev <= gBestFun | mod(GENs,5)==0 % C22
    if errorPrev <= gBestFun | mod(GENs,1)==0 % C2L
%     if errorPrev <= gBestFun  % C2L1
        rc=randperm(Size);        
        F=F(rc); F_t_1=F_t_1(rc);F_t_2=F_t_2(rc);
        E=E(rc,:); E_t_1=E_t_1(rc,:);E_t_2=E_t_2(rc,:);
        pBestS=pBestS(rc,:);pBestFun=pBestFun(rc);    
    end
    
       

    errorPrev = gBestFun;

   
    for region=1:1:numregion,
        % for latest1 - latest6  ***************************************
        [BestFunreg,IndexBestreg]=min(F(startreg(region):endreg(region))); % worse 
        IndexBestreg=IndexBestreg+startreg(region)-1;
        lBestFunreg(region)=BestFunreg;%全局最佳函数值
        lBestSreg(region,:)=E(IndexBestreg(1),:); %全局最佳个体

    end
    
       
    
end % end of "while FES<Max_FES" loop


final_gBestFun(run) = gBestFun;

Time_curve(Max_FES)= cputime - startTime;
Time_curve_res(run,:) = Time_curve;
final_time(run)=cputime - startTime;


final_gBestS(run,:) = gBestS;


% [run exanum]
run
Error=FuncValueCurve(run,Max_FES) %-f_bias

end  % end of "for run=1:1:runs" loop



    filename = ['out_SOAver100205_Size' num2str(Size) '_Iter' num2str(Max_Gens) '_FES' num2str(Max_FES) '_Runs' num2str(runs)];
    save([filename,'.mat'],'final_gBestS','final_gBestFun','final_time','Time_curve_res','Max_Gens','Max_FES','Size','runs','FuncValueCurve','gbest_curve_res');


                             