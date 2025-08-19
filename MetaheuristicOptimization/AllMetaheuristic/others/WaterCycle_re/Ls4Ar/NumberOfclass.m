function count = NumberOfclass(C)
%==========================================================================
[SortC,ind]=sortrows(C);
x=[];
x(1)=1;
for i=2:length(ind)
    if ~isequal(SortC(i-1,:),SortC(i,:))
        x=union(x,i);
    end
end
x(end+1)=length(ind)+1;

count=zeros(1,length(ind));
for i=2:length(x)
    for j=x(i-1):x(i)-1
        count(ind(j))=x(i)-x(i-1);
    end
end

end