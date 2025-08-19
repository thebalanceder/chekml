function [Universe]= ObtainUniverse( C )
%Delete the same object
%Input£ºC
%Ouput£º Universe 
%--------------------------------
[numobj,~]=size(C);
Index=[];
Index(1)=1;
[~,ind]= sortrows(C);
U=ind;
for j=2:numobj
    A=C(U(j),:);
    B=C(U(j-1),:);
    if(~isequal(A,B))
        Index(end+1)=j;
    end
end
Universe=U(Index);
end

