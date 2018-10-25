clear
close all
clc

%% Regression in multi-dimension problems
% This script is general and can be used in all problems

m = 1000;
n = 100;                                 % samples
d = 1;                                   % dimension of the problem
sigma = .05;

XT = linspace(-2*pi, 2*pi, m)';
YT = sinc(XT);

X = rand(n,d)*4*pi - 2*pi;               % calculations to have samples in the same range of the sin [-2pi, 2pi]
Y = sinc(X) + sigma * randn(n,1);

figure, box on, hold on, grid on
plot(XT,YT,'g');
plot(X,Y,'ob');

%% Kernel method
% min_w || X w - y ||^2 + lambda ||w||^2
% X = [phi(x_1)',...,phi(x_n)']
% w = X' * alpha
% alpha = (Q+ lambda * I)^{-1} y
% Q_{i,j} = phi(x_i)' * phi(x_j) = exp(gamma * ||x_i - x_j||^2)
% f(x) = w' phi(x) = sum_{i=1}^n alpha phi(x_i) phi(x)
%                  = sum_{i=1}^n exp(-gamma*||x_i - x||^2)

Q = zeros(n,n);

% I try to tune the values
% I have to split the data into train and test set, but i can't do it only
% one time. I need to repeat hte expeiment some times and find the best
% lambda and gamma on average
nl = round(.7*n);                                   % learning set (70% of data)
nv = n - nl;                                        % validation set (the remaining)
% I can compute the pair-wise distances here
PD = pdist2(X,X);                                    % optimisation step
err = zeros(30*30,1);                               % instead of using a matrix we use a vector to have only a single index
for k = 1:30                                        % 30 is the magical number
    i = randperm(n);
    il = sort(i(1:nl));
    iv = sort(i(nl+1:end));                         % optimisation steps: see QV comment below
    j = 0;
    for gamma = logspace(-4,3,30)
        QL = exp(-gamma * PD(il, il));              % precomputable (if gamma is in the external cycle)
        QV = exp(-gamma * PD(iv, il));              % in these two lines I'm accessing rows and columns of a matrix, where indexes are scrumbled. We can improve the access by sorting il and iv
        % If a matrix is sparse there's a waste of space. In matlab I can
        % transform it using sparse(M) to obtain a sparse representation
        % (memory efficient)
        for lambda = logspace(-4,3,30)
            j = j + 1;
            alpha = (QL + lambda * eye(nl,nl))\Y(il);              
            YP = QV * alpha;
            err(j) = err(j) + mean(abs(YP-Y(iv))) / 30;
        end
    end
end

% Finally I find the best lambda and gamma
j = 0;
err_best = Inf;
for gamma = logspace(-4,3,30)
   for lambda = logspace(-4,3,30)
      j = j + 1;
      if (err(j) < err_best)
          err_best = err(j);
          gamma_best = gamma;
          lambda_best = lambda;
      end
   end
end

Q = exp(-gamma_best * pdist2(X,X));
alpha = (Q + lambda_best * eye(n,n)) \ Y;
YP = exp(-gamma_best * pdist2(XT,X)) * alpha;
plot(XT,YP,'r');