clear
close all
clc

%% Multi-Class classification
nc = 30;                                % number of samples per class
c = 7;                                  % number of classes
r = 5;                                  % radius
X = [];
Y = [];

% We create the training set: a circle of radius r
for i = 1:c
    theta = i * (2 * pi / c);
    X = [X; randn(nc, 1) + r*cos(theta), randn(nc, 1) + r*sin(theta)];
    Y = [Y; i*ones(nc, 1)];
end
[n, d] = size(X);                       % total number of samples and dimension

%Normalization
for i = 1:d
   mi = min(X(:,i));
   ma = max(X(:,i));
   diff = ma - mi;
   X(:,i) = 2 * (X(:,i) - mi) / diff - 1;
end

% Plotting
colors = 'rbcmkyg';
figure, box on, grid on, hold on
for i = 1:c
   plot(X(Y==i,1), X(Y==i,2), ['o', colors(i)]); 
end

%% Implementation of all vs all learning phase
lambda = 1;                             % regularization term
W = [];
% All the classes must compete against all the others, so two nested loops
for i1 = 1:c
   for i2 = i1+1:c
       cp = Y == i1;                    % class plus
       cm = Y == i2;                    % class minus
       all = cp | cm;                   % all the points, labelled as +1 or -1
       Ytmp = cp*1 + cm*(-1);           
       W = [W, (X(all,:)'*X(all,:) + lambda*eye(d))\(X(all,:)'*Ytmp(all))]; % i store all Ws
   end
end

%% Classification phase
ns = 10000;
XS = 2 * rand(ns, d) - 1;
YS = zeros(ns, c * (c-1) / 2);
i3 = 0;                                 % tells which model i'm using in a particular iteration
for i1 = 1:c
   for i2 = i1+1:c
      i3 = i3 + 1;
      YS(:, i3) = XS * W(:, i3);
      % I go back to the original class (1,..,7)
      YS(:, i3) = (YS(:, i3) >= 0)*i1 + (YS(:, i3) < 0)*i2;
   end
end
YS = mode(YS, 2);                       % i take the most frequent results, row-wise

% Plot
for i = 1:c
   plot(XS(YS==i,1), XS(YS==i,2), ['.', colors(i)], 'MarkerSize', 1); 
end

% In the model there is no bias, so all serparators pass through the origin
% If we reduce nc, the classes will not have the same area anymore
% lambda should be chosen not a priori, but reapiting the process for
% different values and selecting the one with the lowest error











