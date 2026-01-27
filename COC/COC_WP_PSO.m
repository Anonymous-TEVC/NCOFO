clear,close,clc;
%% Parameter settings
N=20;  %粒子数目
d=2;
c=3.2;
xmin=-4;
xmax=4;
vmax=1;
vmin=-1;
% f=@(x)f2(x,0)/64.6231+randn*1;   
f=@(x)f4(x,0)/12.5399+randn*1;    %The objective function needs to be normalized, with the maximum value set to 1.
k=20;
G=10000;  %Number of iterations
L=1000;% Number of tests
count=0;
w_ini=0.9;
w_end=0.4;
c1=1.5;
c2=1.5;
%% Initialization
for l=1:L
trace=[];
x=rand(N,d)*(xmax-xmin)+xmin;
v=rand(N,d)*(vmax-vmin)+vmin;
fitness=zeros(1,N);
pbest=x;
pbest_value=ones(N,1)*10000;
gbest=x(1,:);
gbest_value=10000;
for g=1:G
    %     
    for i =1:N
        fitness(i)=f(x(i,:));
        if fitness(i)<f(gbest)
            gbest=x(i,:);
            gbest_value=fitness(i);
        end
    end
    if g>1&&gbest(1)^2+gbest(2)^2<=0.01
        count=count+1;
        break;
    end
    w=(w_ini-w_end)*(10000-g)/10000+w_end;
    %% weighted consensus center
    k=20-floor(g/1000);
    [pass, index] = sort(fitness);
    index=index(1:k);
    elite_value=fitness(index);      %Elite Particle Fitness Set
    elite_x=x(index,:);    %Elite Particle Location Set
    fmax=max(elite_value);
    fmin=min(elite_value);
    m=exp((fmax-elite_value)/(fmax-fmin));
    W=m/(sum(m));
    theta=W*elite_x;
    for i=1:N
        v(i,:)=w*v(i,:)+c*rand*(theta-x(i,:));
        v(i,:)=corrxv1(v(i,:),vmin,vmax);
        x(i,:)=x(i,:)+v(i,:);
    end
    trace(g)=min(fitness);
end
end

fprintf('The probability of success is: %g\n', count / L);


function out = f2(x,p)
%Rastrigin's function:
if p==0
    p=[1,10,10,1,1];
end
if x(1)^2+x(2)^2<=0.01
    out=0;
else
    out=p(2)+p(3)+p(1)*x(1)^2-p(2)*cos(p(4)*2*pi*x(1))+x(2)^2-p(3)*cos(p(5)*2*pi*x(2));
end

end

function out = f4(x,p)
if p==0
    p=[20,1,1,1,1,0.5,0.5];
end
if x(1)^2+x(2)^2<=0.01
    out=0;
else
    out=-p(1)*exp(-0.2*sqrt(p(6)*(p(2)*x(1).^2+p(3)*x(2).^2)))-exp(p(7)*(cos(p(4)*2*pi*x(1))+cos(p(5)*2*pi*x(2))))+p(1)+exp(1);
end
end

function output = corrxv1(input,xvmin,xvmax)
input_shape=size(input);
output=ones(input_shape).*(input>xvmax)*xvmax+ones(input_shape).*(input<xvmin)*xvmin+input.*(input<=xvmax & input>=xvmin);
end