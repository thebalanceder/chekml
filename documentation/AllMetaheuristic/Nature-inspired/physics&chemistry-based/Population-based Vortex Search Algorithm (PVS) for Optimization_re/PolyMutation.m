% Polynomial Mutation
function [child, state] = PolyMutation(child, dim, LB, UB,ProbMut)

    distributionIndex=20;
%     ProbMut=1/dim;
    % ProbMut=0.1;
    % ProbMut=0.1;

    state=0;
    for i=1:dim
        if(rand<=ProbMut)
            y=child(i);
            yL=LB(i);
            yU=UB(i);
            delta1 = (y-yL)/(yU-yL);
            delta2 = (yU-y)/(yU-yL);

            mut_pow=1/(1+distributionIndex);
            rnd=rand;
            if(rnd<=0.5)
               xy=1-delta1;
               val=2*rnd+(1-2*rnd)*xy^(distributionIndex+1);
               deltaq=(val^mut_pow)-1;

            else
                xy=1-delta2;
                val=2*(1-rnd)+2*(rnd-0.5)*xy^(distributionIndex+1);
                deltaq=1-(val^mut_pow);
            end
            y=y+deltaq*(yU-yL);
            if y<yL
                y=yL;
            end
            if y>yU
                y=yU;
            end

            child(i)=y;    
            state=state+1;
        end
    end
    
end

