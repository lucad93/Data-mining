clear 
close all
clc

%%
n = 100;
d = 2;
X = [randn(n/2,d)+2;randn(n/2,d)-2];
Y = [ones(n/2,1);-ones(n/2,1)];

%% Normalizzo
for i = 1:d
    mi = min(X(:,i));
    ma = max(X(:,i));
    di = ma - mi;
    X(:,i) = 2*(X(:,i)-mi)/di-1;
end

%%
figure, hold on, box on, grid on
plot(X(Y<0,1),X(Y<0,2),'ob','MarkerSize',10)
plot(X(Y>0,1),X(Y>0,2),'or','MarkerSize',10)

nl = round(.7 * n);
nv = n - nl;
err = zeros(30,1);
for k = 1:30
    % Divido in training set e validation set
    i  = randperm(n);
    il = sort(i(1:nl));
    iv = sort(i(nl+1:end));
    j = 0;
    
    XL = X(il,:);
    YL = Y(il);
    XV = X(iv,:);
    YV = Y(iv);
    
    for C = logspace(-4,3,30)
        j = j + 1;
        Q = diag(YL)*(XL*XL')*diag(YL);
        [~,err_,alpha,b] = SMO2_ab(nl,Q,-ones(nl,1),YL',zeros(nl,1),C*ones(nl,1),...
                                  1000000,.0001,zeros(nl,1));
        if (err_ ~= 0)
            warning('Problem in SMO')
        end
        w = XL'*diag(YL)*alpha;
        
        YS = XV*w+b;
        YS(YS>=0) = 1;
        YS(YS<0)  = -1;
        err(j) = err(j) + sum(YS ~= YV)/30;
    end
end
j = 0;
err_best = Inf;
for C = logspace(-4,3,30)
   j = j + 1;
   if (err(j) <= err_best)
      err_best = err(j);
      C_best = C;
   end
end
%class_err_best
C = C_best;
Q = diag(Y)*(X*X')*diag(Y);
[~,err_,alpha,b] = SMO2_ab(n,Q,-ones(n,1),Y',zeros(n,1),C*ones(n,1),...
                          1000000,.0001,zeros(n,1));
if (err_ ~= 0)
    warning('Problem in SMO')
end
w = X'*diag(Y)*alpha;

%% Plot the separator
ns = 10000;
XS = 2*rand(ns,d)-1;
YS = XS*w+b;

plot(XS(YS<0,1),XS(YS<0,2),'.c','MarkerSize',1)
plot(XS(YS>0,1),XS(YS>0,2),'.m','MarkerSize',1)
plot(XS(YS<-1,1),XS(YS<-1,2),'.b','MarkerSize',1)
plot(XS(YS>+1,1),XS(YS>+1,2),'.r','MarkerSize',1)

%% Check the solution
plot(X(alpha==0,1),X(alpha==0,2),'sg','MarkerSize',10)
plot(X(alpha>0&alpha<C,1),X(alpha>0&alpha<C,2),'sk','MarkerSize',10)
plot(X(alpha==C,1),X(alpha==C,2),'sy','MarkerSize',10)

%% Compute the duality gap
dualcost = -(.5*alpha'*Q*alpha-ones(n,1)'*alpha);
primalcost = .5*w'*w+C*sum(max(0,1-diag(Y)*(X*w+b))); 
dualitygap = abs(dualcost-primalcost);
title(sprintf('DG: %e   Best error: %e   Best C: %e',dualitygap,err_best,C_best));