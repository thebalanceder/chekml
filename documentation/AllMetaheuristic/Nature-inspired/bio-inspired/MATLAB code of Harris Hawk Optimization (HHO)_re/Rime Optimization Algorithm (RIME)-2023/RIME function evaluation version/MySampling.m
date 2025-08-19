%---------------------------------------------------------------------------------------------------------------------------
% RIME 
% RIME: A physics-based optimization
% Website and codes of RIME:http://www.aliasgharheidari.com/RIME.html

% Hang Su, Dong Zhao, Ali Asghar Heidari, Lei Liu, Xiaoqin Zhang, Majdi Mafarja, Huiling Chen 

%  Last update: Feb 11 2023

%  e-Mail: as_heidari@ut.ac.ir, aliasghar68@gmail.com, chenhuiling.jlu@gmail.com 
%  
%---------------------------------------------------------------------------------------------------------------------------
%  Authors: Ali Asghar Heidari(as_heidari@ut.ac.ir, aliasghar68@gmail.com),Huiling Chen(chenhuiling.jlu@gmail.com) 
%---------------------------------------------------------------------------------------------------------------------------

% After use of code, please users cite to the main paper on RIME:
% Hang Su, Dong Zhao, Ali Asghar Heidari, Lei Liu, Xiaoqin Zhang, Majdi Mafarja, Huiling Chen  
% RIME: A physics-based optimization
%Neurocomputing,ELSEVIER- 2023 
%---------------------------------------------------------------------------------------------------------------------------
% You can also follow the paper for related updates in researchgate: 
% https://www.researchgate.net/profile/Ali_Asghar_Heidari.

%  Website and codes of RIME:%  http://www.aliasgharheidari.com/RIME.html

% You can also use and compare with our other new optimization methods:
                                                                       %(RIME)-2023-http://www.aliasgharheidari.com/RIME.html
																	   %(INFO)-2022- http://www.aliasgharheidari.com/INFO.html
																	   %(RUN)-2021- http://www.aliasgharheidari.com/RUN.html
                                                                       %(HGS)-2021- http://www.aliasgharheidari.com/HGS.html
                                                                       %(SMA)-2020- http://www.aliasgharheidari.com/SMA.html
                                                                       %(HHO)-2019- http://www.aliasgharheidari.com/HHO.html  

%---------------------------------------------------------------------------------------------------------------------------

%---------------------------------------------------------------------------------------------------------------------------

function [ out_ver ] = MySampling(in_ver,num)
in_num=size(in_ver,2);
out_ver=zeros(1,num);
step = round(in_num/num);
if step>0  % 
    out_ver=in_ver(round(linspace(1,in_num,num)));
else      % 
    index=1:1:num;
    newLocalIndex=round(linspace(1,num,in_num));
    restIndex=setdiff(index, newLocalIndex);
    out_ver(newLocalIndex)=in_ver;
    for i=1:size(restIndex,2),
       out_ver(restIndex(i))=out_ver(restIndex(i)-1);
    end
%     out_ver(restIndex)=out_ver(restIndex-1);

end
end

