function z=BenchmarkFunction(x,number)
    if nargin<2
        error('Name or Number of function is not specified.');
    end
    
    switch number
        case 1
            z=sum(x'.^2)';
        case 2
            z=zeros(size(x,1),1);
            for i=1:size(x,2);
                z=z+i*x(:,i).^4;
            end
        case 3
            z=0.5+(sin(sqrt(x(:,1).^2+x(:,2).^2))-0.5)./(1+0.001*(x(:,1).^2+x(:,2).^2)).^2;
        case 4
            z=zeros(size(x,1),1);
            p=ones(size(x,1),1);
            for i=1:size(x,2)
                z=z+1/4000*(x(:,i)-100).^2;
                p=p.*(cos((x(:,i)-100)/sqrt(i))+1);
            end
        case 5
            z=zeros(size(x,1),1);
            for i=1:size(x,2)-1;
                z=z+100*(x(:,i+1)-x(:,i).^2).^2+(x(:,i)-1).^2;
            end
        case 6
            z=sum(x'.^2-10*cos(2*pi*x')+10)';
        otherwise
            error('Invalid function number is used.');
    end
    
end