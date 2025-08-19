function [num_iop] = iop(C,D)
%UNTITLED3 ´Ë´¦ÏÔÊ¾ÓÐ¹Ø´Ëº¯ÊýµÄÕªÒª
%   ´Ë´¦ÏÔÊ¾ÏêÏ¸ËµÃ÷
[obj,~]=size(D);
if isempty(C)
    eqclass=zeros(1,obj)+obj;
    count=eqclass-Equivalentclass( [C,D] );
else
    count= Equivalentclass( C )- Equivalentclass( [C,D] );
   
end
 num_iop=sum(count);
end

