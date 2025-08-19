function [red] = fastRed(C,D)
% A Fast Method for Constructing Relative Reduct
%
[~,att]=size(C);
iopC=iop(C,D);
w=zeros(1,att);
for i=1:att
    w(i)=iop(C(:,i),D);
end
[~,ind]=sort(w);
red=[];
for i=1:att
    red=union(red,ind(i));
    if iop(C(:,red),D)==iopC
        break;
    end
end
%%
% cd=red;
%     while ~isempty(cd)
%         newpos=ObtainUniverse(( C(:,red) ));
%         [~,indcd]=max(w(cd));
%          redcd=setdiff(red,cd(indcd));
%         cd=setdiff(cd,cd(indcd));
%         if iop( C(newpos,redcd),D(newpos,:))==iop( C(newpos,red),D(newpos,:))
%             red=redcd;
%         end
%     end
% end

