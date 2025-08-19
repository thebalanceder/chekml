function [count]= Equivalentclass( C )
%Compute the index of equivalent class
%Input£ºC
%Ouput£ºthe index of equivalentclass 
%--------------------------------
[numOfobject,~]=size(C);
Index=[];
Index(1)=1;
[~,ind]= sortrows(C);
Obj=ind;
U=ind;
for j=2:numOfobject
    A=C(U(j),:);
    B=C(U(j-1),:);
    if(~isequal(A,B))
        Index(end+1)=j;
    end
end
%Index(end+1)=length(U);
[numobj,~]=size(C);
count=zeros(1,numobj);
for i=1:length(Index)-1
    for j=Index(i):Index(i+1)-1
        count(Obj(j))=Index(i+1)-Index(i);
    end
end
for k=Index(end):numobj
    count(Obj(k))=numobj-Index(end)+1;
end
end

