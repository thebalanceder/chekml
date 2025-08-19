function [u,v] = APSmechanism(C,D,B)
%Attribute pair selection mechanism
% u=0 and v=0 indicate that the required attribute pair has not been found.
%%
u=0;
v=0;
[~,numatt]=size(C);
pos=PositiveRegion(C,D);
unpos=setdiff(pos,PositiveRegion(C(:,B),D));
unred=setdiff(1:numatt,B);
add=[];
for k=1:length(unred)
    if length(PositiveRegion(C(unpos,unred(k)),D(unpos,:)))==length(unpos)
        add=union(add,unred(k));
    end
end

for i=1:length(add)
    U=ObtainUniverse( C(:,union(B,add(i))) );
    for j=1:length(B)
        testB=setdiff(union(B,add(i)),B(j));
        if length(PositiveRegion(C(U,testB),D(U,:)))==length(U)
            v=add(i);
            u=B(j);
            break;
        end
    end
end

end

