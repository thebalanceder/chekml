function [ PosSet ] =  PositiveRegion( C,D )
%Compute  positiveRegion
%  Input£º C and D
%  Output£º POSc(D)
if isempty(C)
    PosSet=[];
else
    count= Equivalentclass( C )- Equivalentclass( [C,D] );
    PosSet=find(count==0);
end

end




