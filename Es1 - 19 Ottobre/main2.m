clear;
close all;
clc;

%% Regression in one dimensional space

%% Finding the perfect p depending on the number of sapmples
m = 1000;                       % number of points to plot the true function
sigma = .05;                    % variance of the noise (gaussian distrib) => the most important variable

% True function
XT = linspace(0,1,m)';
YT = XT.^2;

% I delete warnings about badly scaled matrixes
[~, MSGID] = lastwarn();
warning('off', MSGID);

for n = [3,6,10,15,30]
    X = rand(n,1);
    Y = X.^2 + sigma * randn(n,1);
    AL = [];
    AT = [];
    i2 = 0;
    for p = 0:5
        AL = [AL, X.^p];
        AT = [AT, XT.^p];
        c = (AL'*AL)\(AL'*Y);
        YP = AT*c;
        err = mean(abs(YT-YP));
        fprintf('%.3e ',err);
    end
    fprintf('\n');
end

% This values change because there's noise
% Now to make sure that these values are correct i repeat 30 times the
% experiment and take the mean errors. 30 times because the variance
% diminuishes with the sqrt(1/n).