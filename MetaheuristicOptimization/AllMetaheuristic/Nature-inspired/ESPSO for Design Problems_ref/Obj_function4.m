%______________________________________________________________________________________
% cantilever beam design problem
%______________________________________________________________________________________

function o=Obj_function4(x)
o = [0];
% 
 o(1)=0.0624*(x(1) + x(2) + x(3) + x(4) + x(5));
%  o(2)=65856000/(30*10^6*x(4)*x(3)^3);
o=o+getnonlinear(x);

function Z=getnonlinear(x)
Z=0;
% Penalty constant
lam=10^10;

g(1)=(61/(x(1)^3)) + (37/(x(2)^3)) + (19/(x(3)^3)) + (7/(x(4)^3)) + (1/(x(5)^3)) - 1;




% No equality constraint in this problem, so empty;
geq=[];

% Apply inequality constraints
for k=1:length(g),
    Z=Z+ lam*g(k)^2*getH(g(k));
end
% Apply equality constraints
for k=1:length(geq),
   Z=Z+lam*geq(k)^2*getHeq(geq(k));
end

% Test if inequalities hold
% Index function H(g) for inequalities
function H=getH(g)
if g<=0,
    H=0;
else
    H=1;
end
% Index function for equalities
function H=getHeq(geq)
if geq==0,
   H=0;
else
   H=1;
end
% ----------------- end ------------------------------

