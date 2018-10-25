clear
close all
clc

%% Regression in multi-dimension problems
% The following example is mono variable to be able of drawing graphs.
m = 1000;
n = 100;                                 % samples
d = 1;                                   % dimension of the problem
sigma = .05;                             % variance of the noise

%If we have large noise value, then we need more samples and taking the
%average value we are able to find the original signal.

XT = linspace(-2*pi, 2*pi, m)';
YT = sinc(XT);

X = rand(n,d)*4*pi - 2*pi;               % calculations to have samples in the same range of the sin [-2pi, 2pi]
Y = sinc(X) + sigma * randn(n,1);

figure, box on, hold on, grid on
plot(XT,YT,'g');
plot(X,Y,'ob');

% NOTE: for cycles and if statements are deprecated in matlab language,
% because they get the code slower.

%% Kernel method
% min_w || X w - y ||^2 + lambda ||w||^2
% X = [phi(x_1)',...,phi(x_n)']
% w = X' * alpha
% alpha = (Q+ lambda * I)^{-1} y
% Q_{i,j} = phi(x_i)' * phi(x_j) = exp(gamma * ||x_i - x_j||^2)
% f(x) = w' phi(x) = sum_{i=1}^n alpha phi(x_i) phi(x)
%                  = sum_{i=1}^n exp(-gamma*||x_i - x||^2)

% First we do this with for loops (slow naive version)
Q = zeros(n,n);
% for i = 1:n
%    for j = i:n
%       tmp = X(i,:) - X(j,:);
%       tmp = tmp * tmp';                        % for the squared norm
%       Q(i,j) = exp(-gamma * tmp);
%       Q(j,i) = Q(i,j);                         % since Q is symmetric I can do this
%       % Matrixes are stored like in Fortran, by columns
%       % So the difference between the j = 1:n and making all computation
%       % and this version using the symmetry is that this version is not
%       % twice as fast
%    end
% end

% Correct way to do it, with computation of f(x)
lambda = .1;                                    % big: more flat (i try to regularize the function); small: solution will be more complex                                     
gamma = 1;                                      % small: more flat line; big: line passes through all points (very non-linear). This because gamma works like the inverse of the variance, where the gaussian is centered in every point. For gamma->Inf we have a dirac's delta
Q = exp(-gamma * pdist2(X,X));                  % point-wise distance of each element. O(n^2 * d)
alpha = (Q + lambda * eye(n,n))\Y;              % O(n^2) cost of this (= cost of the n x n matrix inversion). This is important because we have to compute this lots of times
YP = exp(-gamma * pdist2(XT,X)) * alpha;        % optimal way to do it. O(m*n*d + n*m) = O(m*n*d). I don't like n in this complexity (major drawback of this algorithm)
% Case lambda = 0 and gamma very small: no regularization = the solution
% will pass through all points, no matter the value of gamma. All the
% exp() will go to 1, so the difference between the values will be very
% small

% Naive way to compute f(x)
% YP = zeros(m,1);
% for i = 1:m
%    for j = 1:n
%        tmp = XT(i,:) - X(j,:);
%        tmp = tmp * tmp';
%        YP(i) = YP(i) + alpha(j) * exp(-gamma * tmp);
%    end
% end

plot(XT,YP,'r');