clear; close all; clc;

%% Parameter settings
N    = 20;      % Number of particles
d    = 2;       % Dimension
xmin = -4;
xmax =  4;
vmax =  1;
vmin = -1;

% Noise objective function
f = @(x) f2(x,0)/64.6231 + randn*0.5;  %Normalize the objective function so that its maximum value is 1.
% f=@(x)f4(x,0)/12.5399+randn*0.5;

Gmax  = 10000;  % Maximum number of iterations
L     = 100;    % Number of tests
count = 0;      % Success Count Statistics

a = 1.0;        
b = 3.2;        
K = 2;

rng('shuffle');

%% Multiple experiments
for l = 1:L
    fprintf('Run %d / %d\n', l, L);

    % 初始化
    x = rand(N,d) * (xmax - xmin) + xmin;      % Location
    v = rand(N,d) * (vmax - vmin) + vmin;      % speed

    fitness     = zeros(1,N);
    gbest       = x(1,:);
    gbest_value = inf;   

    trace = zeros(1, Gmax);

    for g = 1:Gmax

        %% 1. Calculate fitness & update global optimum
        for i = 1:N
            fitness(i) = f(x(i,:));

            if fitness(i) < f(gbest)
                gbest       = x(i,:);
                gbest_value = fitness(i);
            end
        end

        % Determine if we have reached the vicinity of the global optimum.
        if g > 1 && (gbest(1)^2 + gbest(2)^2) <= 0.01
            count = count + 1;
            fprintf('  Hit global basin at iter %d\n', g);
            break;
        end

        %% 2. Search Center c
        k = 20 - floor(g / 1000);
        k = max(1, min(N, k));

        [~, index]  = sort(fitness);   
        index       = index(1:k);      
        elite_value = fitness(index);  
        elite_x     = x(index,:);     

        fmax = max(elite_value);
        fmin = min(elite_value);

        if fmax == fmin
            W = ones(1,k) / k;
        else
            m_theta = exp((fmax - elite_value) / (fmax - fmin));
            W       = m_theta / sum(m_theta);
        end

        c = W * elite_x;               

        %% 3. Speed ​​/ Location Update
        for i = 1:N
            r1 = rand;      
            r2 = rand;     

            v(i,:) = a * r1 * v(i,:) ...
                   + (K * b * r2 / 2) * (c - x(i,:));

            v(i,:) = corrxv1(v(i,:), vmin, vmax);

            x(i,:) = x(i,:) + v(i,:);
        end

        trace(g) = min(fitness);
    end
end

success_rate = count / L;
fprintf('\nCOC-GSA  success rate = %.4f  (%d / %d)\n', success_rate, count, L);

figure;
plot(trace(trace ~= 0), 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Best fitness');
grid on;


function output = corrxv1(input,xvmin,xvmax)
input_shape=size(input);
output=ones(input_shape).*(input>xvmax)*xvmax+ones(input_shape).*(input<xvmin)*xvmin+input.*(input<=xvmax & input>=xvmin);
end


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
