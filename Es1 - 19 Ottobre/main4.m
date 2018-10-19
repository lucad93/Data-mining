clear;
close all;
clc;

%% Regression in one dimensional space

%% Data initialization
n = 10;                         % number of samples
m = 1000;                       % number of points to plot the true function
sigma = .05;                    % variance of the noise (gaussian distrib) => the most important variable
p = 9;                          % degree of the polynomial

% True function
XT = linspace(0,1,m)';
YT = XT.^2;

% Estimate y = x^2 where x \in [0,1]
X = rand(n,1);
Y = X.^2 + sigma * randn(n,1);  % i add noise

figure, grid on, box on, hold on
xlim([0,1]);
ylim([0,1]);
plot(X,Y,'ob');                          % i want to put blue (b) balls (o)
plot(XT,YT,'-g');                        % green (g) straight lines (-)

%% Polynomial regression with regularization
lambda = 0.01;
A = [];
for i = 0:p
   A = [A, X.^i]; 
end
% optimal coefficients: c = (A' A + lambda)^+ A'y
c = (A'*A + lambda*eye(size(A'*A)))\(A'*Y);

% I plot the function that i learned
A = [];
for i = 0:p
   A = [A, XT.^i]; 
end
YP = A*c;                          % prediction

% I compute the mean error and set it as the plot title
error = mean(abs(YP-YT));
title(sprintf('err: %e',error));

plot(XT, YP, '-r');                % red lines

% I see that the polynomial is a 2nd order-like, because there's lambda
% which regularizes and limits the dimension on fthe solution (try with
% lambda = 0 and = 1000)
% How cah I check which is the best lambda for my data?
% For small n: lambda->inf is better
% For high n: lambda->0 is better
% With more noise it's better having a bigger lambda (do graph in report)