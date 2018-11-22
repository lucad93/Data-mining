clear;
close all;
clc;

%% Regression in one dimensional space

%% Finding the perfect p depending on the number of sapmples
m = 1000;                       % number of points to plot the true function
sigma = 10;                    % variance of the noise (gaussian distrib) => the most important variable

% True function
XT = linspace(0,1,m)';
YT = XT.^2;

% I delete warnings about badly scaled matrixes
[~, MSGID] = lastwarn();
warning('off', MSGID);

error = zeros(5,6);                             % n x p
for k = 3000
    i1 = 0;
    for n = [3,6,10,15,30]
        i1 = i1 + 1;
        X = rand(n,1);
        Y = X.^2 + sigma * randn(n,1);
        AL = [];
        AT = [];
        i2 = 0;
        for p = 0:5
            i2 = i2 + 1;
            AL = [AL, X.^p];
            AT = [AT, XT.^p];
            c = (AL'*AL)\(AL'*Y);
            YP = AT*c;
            error(i1,i2) = error(i1,i2) + mean(abs(YT-YP));
        end
    end
end
error = error / 3000;
% In the previous line, if we increase the denominator, we note that the
% system behaves differently. If we increase 'p' parameter, we note that
% the variance value increases too. If we use a higher degree regressor,
% the system depends more on the value of the original random variable,
% because of the number of data.

for i = 1:size(error,1)
   for j = 1:size(error,2)
      fprintf('%.3e \t',error(i,j)); 
   end
   fprintf('\n');
end

% variance * sqrt(1/n).
% From left to right things changes even if n=30 (or 3000) is fixed. This because the
% variance changes when p changes. If I use a more complex regressor the
% result will much more depend on the original random variable, there will
% be no smoothness => the variance of the random variable will increase.

% From up to down the down values are more stable. The top values changes a
% lot because i'm sampling less data => the variance will be higher

% The best solutions seems to be p = 2.
% If I increase the noise the best solution will be simpler (p closer to 0)