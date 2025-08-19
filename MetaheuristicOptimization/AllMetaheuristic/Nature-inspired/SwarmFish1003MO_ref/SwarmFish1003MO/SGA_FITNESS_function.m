function [ fitness ] = SGA_FITNESS_function(n, wc, H, va)

% /*M-FILE FUNCTION SGA_FITNESS_function MMM SGALAB */ %
% /*==================================================================================================
%  Simple Genetic Algorithm Laboratory Toolbox for Matlab 7.x
%
%  Copyright 2011 The SxLAB Family - Yi Chen - leo.chen.yi@gmail.com
% ====================================================================================================
%
%File description:
%
% [1]for both single objective and multi objective problems,
%
% [2]accept stand-alone variables and matrix variables, e.g.
%
% (1) stand-alone variables, fitness(x,y,z)
%
% (2) matrix variables,
%
% fitness([x,y,z;x,y,z],[x,y,z;x,y,z],[x,y,z;x,y,z])
%
%Input:
%          User define-- in the format ( x1, x2, x3,... )
%
%Output:
%          fitness--     is the fitness value
%
%Appendix comments:
%
% 02-Dec-2009   Chen Yi
% obsolete SGA__fitness_MO_evaluating.m
%          SGA_FITNESS_MO_function.m
% use      SGA__fitness_evaluating.m
%          SGA_FITNESS_function.m (for both single objective and multi
%                                  objective problems )
%
%Usage:
%     [ fitness ] = SGA_FITNESS_function( xi,... )
%===================================================================================================
%  See Also:
%
%===================================================================================================
%
%===================================================================================================
%Revision -
%Date        Name    Description of Change email                 Location
%27-Jun-2003 Chen Yi Initial version       chen_yi2000@sina.com  Chongqing
%14-Jan-2005 Chen Yi update 1003           chenyi2005@gmail.com  Shanghai
%02-Dec-2009 Chen Yi obsolete
%                     SGA__fitness_MO_evaluating.m
%                     SGA_FITNESS_MO_function.m, use
%                     SGA_FITNESS_function for both single and multi
%                     objective problems
%HISTORY$
%==================================================================================================*/

%SGA_FITNESS_function begin

% n --x(1),Number of channels, ͨ����,        [4,20];
% wc--x(2),Width of channel, ������� (mm),   [1e-3,4e-3];
% V --x(3),Velocity of inlet, ����ٶ� (m/s), [0,2];
% h --x(4),Height of channel, ɢ�����߶� (mm),[2e-3,5e-3];

%n, wc, H, va
x = zeros(1,4);

x(1) = n;
x(2) = wc;
x(3) = va;
x(4) = H;

%  minimize a multicomponent objective function.
objs = Rtotal_multi(x);

%Ŀ�꺯��1,ɢ��������
fitness(1) = 1./(objs(1) + eps);

% %Ŀ�꺯��2��ɢ������ѹ����ʧ
% fitness(2) = 1./(objs(2) + eps);

%Ŀ�꺯��3, ��ŵ��
fitness(3) = 1./(objs(3) + eps);

% % %Ŀ�꺯��1,ɢ��������
% % % fitness(1) = objs(1)*100;
% 
% Ŀ�꺯��2��ɢ������ѹ����ʧ
fitness(2) = objs(2)*10;
% 
% %Ŀ�꺯��3, ��ŵ��
% fitness(3) = objs(3);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  minimize a multicomponent objective function.
function fitness = Rtotal_multi(x)

% n --x(1),Number of channels, ͨ����,        [4,20];
% wc--x(2),Width of channel, ������� (mm),   [1e-3,4e-3];
% V --x(3),Velocity of inlet, ����ٶ� (m/s), [0,2];
% h --x(4),Height of channel, ɢ�����߶� (mm),[2e-3,5e-3];

L=32e-3;  %ɢ�������
vn=1e-6;  %ˮ����
rho=1000; %ˮ���ܶ�
kf=0.595; %ˮ���ȴ���ϵ��
tb=1e-3;  %ɢ�����װ�ĺ��
ka=160;   %ɢ�����ĵ���ϵ��
Pr=7;     %�����س���
cp=4183;  %ˮ������

wf=(L-x(1)*x(2))/(x(1)+1);%��Ƭ�����ˮ����ȵĹ�ϵ
At=L^2;%�װ����
Ab=x(2)*(L-2*wf);%ˮ���������
Aw=x(4)*(L-2*wf-x(2));%ˮ���������
Dh=2*x(2)*x(4)/(x(2)+x(4));%����ˮ��ֱ��
Re=x(3)*Dh/vn;%��ŵ��
G=((x(2)/x(4))^2+1)/((x(2)/x(4))+1)^2;%G
Nu1=8.31*G-0.02;%��ȫ��չ��Ŭ������
L1=Re*Dh*Pr/(L-wf);%x*
Nu=((2.22*L1^0.33)^3+Nu1^3)^(1/3);%ƽ��Ŭ������
hf=Nu*kf/Dh;%����ϵ��

R1=tb/(ka*At);%����R1
m=sqrt(hf/(ka*wf));%mֵ
enta=tanh(m*x(4))/(m*x(4));%��ƬЧ��
As=x(1)*Ab+2*Aw*x(1)*enta;%��Ч�������

R2=1/(hf*As);%����R2
Rt=R1+R2;%����ģ�͵�������

fitness(1)=Rt+1/(x(4)*x(2)*x(3)*rho*cp);%Ŀ�꺯��1,ɢ��������



a=x(4)/x(2);%��߱�
b=((L-x(1)*x(2))/(x(1)+1))/x(2);%��Ƭ��ͨ�����֮��
L2=Re*Dh/(L-wf);%��ڶγ��Ȳ���
fl=(19.64*G+4.7)/Re;%��ȫ��չ�Ĳ�����ʧϵ��
f=((3.2*L2^0.57)^2+(fl*Re)^2)^0.5/Re;%ƽ����ʧϵ��
if a>=1&&b<=2
    if Re>=1000
        fw=8.09*(1-0.3439*a+0.042*a^2)*(1-0.3315*b+0.1042*b^2);%�����ʧϵ��
    else
        fw=100000;
    end
else
    if Re<1000
        fw=0.46*Re^(1/3)*(1-0.2*a+0.0022*a^2)*(1+0.26*b^(2/3)-0.0018*b^2);
    else
        fw=3.8*(1-0.1*a+0.0063*a^2)*(1+0.12*b^(2/3)-0.0003*b^2);
    end
end
pz1=2*f*(L-wf)*rho*x(3)^2/Dh;%ֱ�ܵ�ѹ����ʧ
pz2=2*f*(L-2*wf)*rho*x(3)^2/Dh;
pw=0.5*fw*rho*x(3)^2;%��ܵ�ѹ����ʧ
P=2*pz1+(x(1)-2)*pz2+(x(1)-1)*pw;%��ѹ��

fitness(2)=P;%Ŀ�꺯��2��ɢ������ѹ����ʧ

fitness(3)=max(0,0.0002-wf)+max(0,Re-2300);