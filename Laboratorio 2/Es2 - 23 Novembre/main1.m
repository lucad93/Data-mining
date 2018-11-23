clear
close all
clc
 
%% Decision Trees
nc = 30;
c = 7;
r = 5;
X = [];
Y = [];
for i = 1:c
    theta = i*(2*pi/c);
    X = [X; randn(nc,1)+r*cos(theta), randn(nc,1)+r*sin(theta)]; %#ok<AGROW>
    Y = [Y; i*ones(nc,1)]; %#ok<AGROW>
end
[n, d] = size(X);
 
%%
for i = 1:d
    mi = min(X(:,i));
    ma = max(X(:,i));
    di = ma - mi;
    X(:,i) = 2*(X(:,i)-mi)/di-1; %#ok<SAGROW>
end
 
%%
colors = 'rbcmkyg';
figure, box on, grid on, hold on
for i = 1:c
    plot(X(Y==i,1),X(Y==i,2),['o',colors(i)],'MarkerSize',10,'linewidth',5)
end
 
 
%%
depth=7;
T = DT_learn(X,Y,depth);
 
%%
ns = 10000;
XS = 2*rand(ns,d)-1;
YS = DT_forw(T,XS);
 
%%
for i = 1:c
    plot(XS(YS==i,1),XS(YS==i,2),['.',colors(i)],'MarkerSize',1)
end