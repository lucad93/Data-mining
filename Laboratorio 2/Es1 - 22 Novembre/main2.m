clear
close all
clc

%% Support Vector Machines
n = 100;
d = 2;
X = [randn(n/2, d) + 2; randn(n/2, d) - 2];
Y = [ones(n/2, 1); -ones(n/2, 1)];

%Normalization
for i = 1:d
   mi = min(X(:,i));
   ma = max(X(:,i));
   diff = ma - mi;
   X(:,i) = 2 * (X(:,i) - mi) / diff - 1;
end

figure, hold on, box on, grid on
plot(X(Y<0, 1), X(Y<0, 2), 'ob');
plot(X(Y>0, 1), X(Y>0, 2), 'or');

%% Learning phase
% dual formulation: min_alpha .5 * alpha' Q alpha - 1 alpha
% s.t. y'alpha = 0, 0 <= alpha <= C, C has the opposite meaning of lambda
% Q_ij = y_i y_j x'_i x_j
% We use quadprog to solve this problem, which solves quadratic programming
% problems, even if the matlab algorithm doesn't work very well for SVMs
C = 1;
Q = diag(Y) * (X * X') * diag(Y);             % waste of memory but it's faster than two nested loops
alpha = quadprog(Q, -ones(n, 1), [], [], Y', 0, zeros(n, 1), C*ones(n, 1));
w = X' * diag(Y) * alpha;
% The bias is computed only for alpha in (0, C)
i = find(alpha > 0+1e-5 | alpha < C-(+1e-5)); i = i(1);
b = Y(i) - w' * X(i, :)';

%% Classification phase
ns = 10000;
XS = 2 * rand(ns, d) - 1;
YS = XS * w + b;

% Plot
plot(XS(YS<0, 1), XS(YS<0, 2), '.c', 'MarkerSize', 1);
plot(XS(YS>0, 1), XS(YS>0, 2), '.m', 'MarkerSize', 1);
plot(XS(YS<-1, 1), XS(YS<-1, 2), '.b', 'MarkerSize', 1);
plot(XS(YS>+1, 1), XS(YS>+1, 2), '.r', 'MarkerSize', 1);

% I plot the correct points and the ones that lay on the support hyperplanes
plot(X(alpha == 0, 1), X(alpha == 0, 2), 'sg');
plot(X(alpha > 0 & alpha < C, 1), X(alpha > 0 & alpha < C, 2), 'sk');
plot(X(alpha == C, 1), X(alpha == C, 2), 'sy');

% Primal and dual costs
dualcost = -(.5 * alpha' * Q * alpha - ones(n, 1)' * alpha);
primalcost = .5 * w' * w + C * sum(max(0, 1-diag(Y)*(X*w + b)));
dualitygap = abs(dualcost - primalcost);
title(sprintf('Duality Gap: %e', dualitygap));