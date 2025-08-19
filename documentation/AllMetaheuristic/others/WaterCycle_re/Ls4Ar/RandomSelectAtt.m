function [a] = RandomSelectAtt(B)
%UNTITLED3 ´Ë´¦ÏÔÊ¾ÓÐ¹Ø´Ëº¯ÊýµÄÕªÒª
%   ´Ë´¦ÏÔÊ¾ÏêÏ¸ËµÃ÷
if isempty(B)
    disp("Error input")
else
    temp=randperm(length(B),1);
    a=B(temp);
end

