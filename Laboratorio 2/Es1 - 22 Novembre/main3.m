clear 
close all
clc
 
%%
n = 100;
d = 2;
X = [randn(n/2,d)+2;randn(n/2,d)-2];
Y = [ones(n/2,1);-ones(n/2,1)];
 
%%
for i = 1:d
    mi = min(X(:,i));
    ma = max(X(:,i));
    di = ma - mi;
    X(:,i) = 2*(X(:,i)-mi)/di-1;
end
 
%%
figure, hold on, box on, grid on
plot(X(Y<0,1),X(Y<0,2),'ob','MarkerSize',10,'linewidth',5)
plot(X(Y>0,1),X(Y>0,2),'or','MarkerSize',10,'linewidth',5)
%%
C = 1;
Q = diag(Y)*(X*X')*diag(Y);
[~,err,alpha,b] = SMO2_ab(n,Q,-ones(n,1),Y',zeros(n,1),C*ones(n,1),...
                          1000000,.0001,zeros(n,1));
if (err ~= 0)
    warning('Problem in SMO')
end
w = X'*diag(Y)*alpha;
 
%%
ns = 10000;
XS = 2*rand(ns,d)-1;
YS = XS*w+b;
 
%%
plot(XS(YS<0,1),XS(YS<0,2),'.c','MarkerSize',1)
plot(XS(YS>0,1),XS(YS>0,2),'.m','MarkerSize',1)
plot(XS(YS<-1,1),XS(YS<-1,2),'.b','MarkerSize',1)
plot(XS(YS>+1,1),XS(YS>+1,2),'.r','MarkerSize',1)
 
%%
plot(X(alpha==0,1),X(alpha==0,2),'sg','MarkerSize',10,'linewidth',5)
plot(X(alpha>0&alpha<C,1),X(alpha>0&alpha<C,2),'sk','MarkerSize',10,'linewidth',5)
plot(X(alpha==C,1),X(alpha==C,2),'sy','MarkerSize',10,'linewidth',5)

%%
dualcost = -(.5*alpha'*Q*alpha-ones(n,1)'*alpha);
primalcost = .5*w'*w+C*sum(max(0,1-diag(Y)*(X*w+b))); 
dualitygap = abs(dualcost-primalcost);
title(sprintf('DG: %e',dualitygap));