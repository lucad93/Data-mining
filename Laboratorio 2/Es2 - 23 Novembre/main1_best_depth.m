clear
close all
clc
 
%% Decision Trees
nc = 30;
c = 2;
r = 5;
X = [];
Y = [];
dataset = readtable('Iris.csv', 'Delimiter', ',', 'ReadVariableNames', true);
labels = table2array(dataset(:,6));
data = table2array(dataset(:,2:5));
% classesS = strings(150,1);
classes = zeros(150,1);
for i=1:150
    if labels(i) == "Iris-setosa"
        classes(i) = 1.0;
    else
        classes(i) = -1.0;
    end
end


% for i = 1:c
%     theta = i*(2*pi/c);
%     X = [X; randn(nc,1)+r*cos(theta), randn(nc,1)+r*sin(theta)]; %#ok<AGROW>
%     Y = [Y; i*ones(nc,1)]; %#ok<AGROW>
% end
X = [data(:,1) data(:,3:4)];
Y = classes;

[n, d] = size(X);
nl = round(.7*n);                                   % learning set (70% of data)
i = randperm(n);
il = sort(i(1:nl));                             % sort function to get the result less disordered
iv = sort(i(nl+1:end));                         % optimisation steps: see QV comment below 
%%
for i = 1:d
    mi = min(X(:,i));
    ma = max(X(:,i));
    di = ma - mi;
    X(:,i) = 2*(X(:,i)-mi)/di-1;
end

%%
colors = 'rbcmkyg';
figure, box on, grid on, hold on
for i = -1:2:c
    plot3(X(Y==i,1),X(Y==i,2),X(Y==i,3),['o',colors(i+2)]);
end
 
 
%%
for depth = 1:7
    T = DT_learn(X,Y,depth);
end


 
%%
ns = 10000;
XS = 2*rand(ns,d)-1;
YS = DT_forw(T,XS);
 
%%
for i = -1:2:c 
    plot3(XS(YS==i,1),XS(YS==i,2),XS(YS==i,3),['.',colors(i+2)],'MarkerSize',1)
end