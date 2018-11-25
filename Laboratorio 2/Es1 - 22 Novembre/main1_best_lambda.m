clear
close all
clc

%% Multi-Class classification
nc = 80;                                % number of samples per class
c = 4;                                  % number of classes
r = 4;                                  % radius
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

%% Implementation of all vs all
nl = round(.7 * n);                     % learning set dimension = 70% of total data
nv = n - nl;                            % validation set dimension = the remaining data

err = zeros(30, 1);                     % error vector (every value is for a different value of lambda)
for k = 1:30
    i = randperm(n);                    % vector of random data indexes
    il = sort(i(1:nl));                 % vector of learning data indexes
    iv = sort(i(nl+1:end));             % vector of validation data indexes
    j = 0;                              % index for the error vector
    
    % Definition of learning and validation sets
    XL = X(il,:);
    YL = Y(il);
    XV = X(iv,:);
    YV = Y(iv);
    
    for lambda = logspace(-4,3,30)      % regularization term
        W = [];
        j = j + 1;
        
        % Learning phase: all the classes must compete against all the others, so two nested loops
        for i1 = 1:c
           for i2 = i1+1:c
               cp = YL == i1;                % class plus
               cm = YL == i2;                % class minus
               all = cp | cm;               % all the points, labelled as +1 or -1
               Ytmp = cp*1 + cm*(-1);           
               W = [W, (XL(all,:)'*XL(all,:) + lambda*eye(d))\(XL(all,:)'*Ytmp(all))];
           end
        end

        % Classification phase
        YS = zeros(nv, c * (c-1) / 2);          % vector of predicted labels
        i3 = 0;                                 % tells which model i'm using in a particular iteration
        for i1 = 1:c
           for i2 = i1+1:c
              i3 = i3 + 1;
              YS(:, i3) = XV * W(:, i3);
              % I go back to the original class
              YS(:, i3) = (YS(:, i3) >= 0)*i1 + (YS(:, i3) < 0)*i2;
           end
        end
        YS = mode(YS, 2);                       % i take the most frequent results, row-wise
        err(j) = err(j) + sum(YS ~= YV)/30;     % err(j) = average (on k) of classification errors
    end
end

% I find the best lambda
j = 0;
err_best = Inf;
for lambda = logspace(-4,3,30)
   j = j + 1;
   if (err(j) < err_best)
      err_best = err(j);
      lambda_best = lambda;
   end
end

% I re-compute the model W using the best lambda and all data
W = [];
j = j + 1;
for i1 = 1:c
   for i2 = i1+1:c
       cp = Y == i1;
       cm = Y == i2;
       all = cp | cm;
       Ytmp = cp*1 + cm*(-1);           
       W = [W, (X(all,:)'*X(all,:) + lambda_best*eye(d))\(X(all,:)'*Ytmp(all))];
   end
end

% Plotting the data set
colors = 'bygkcmr';
figure, box on, grid on, hold on
for i = 1:c
   plot(X(Y==i,1), X(Y==i,2), ['o', colors(i)]); 
end

% Using the model W with the best lambda to plot a lot of points
% to see the lines
ns = 10000;
XS = 2 * rand(ns, d) - 1;
YS = zeros(ns, c * (c-1) / 2);
i3 = 0;
for i1 = 1:c
   for i2 = i1+1:c
      i3 = i3 + 1;
      YS(:, i3) = XS * W(:, i3);
      YS(:, i3) = (YS(:, i3) >= 0)*i1 + (YS(:, i3) < 0)*i2;
   end
end
YS = mode(YS, 2);
for i = 1:c
   plot(XS(YS==i,1), XS(YS==i,2), ['.', colors(i)], 'MarkerSize', 1); 
end

% Print the best lambda and the best error on the figure
title(sprintf("Best error: %e, Best lambda: %e", err_best, lambda_best));