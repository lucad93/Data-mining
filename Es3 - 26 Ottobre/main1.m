clear
close all
clc
 
%%
D = load('wine_red.csv');
 
%%
for i = 1:size(D,2)
    mi = min(D(:,i));
    ma = max(D(:,i));
    di = ma - mi;
    if (di > 1e-6)
        % D(:,i) = (D(:,i)-mean(D(:,i)))/std(D(:,i)); % 0 mean, 1 std
        % D(:,i) = (D(:,i)-mi)/di; % [0,1]
        D(:,i) = 2*(D(:,i)-mi)/di-1; % [-1,1]
    else
        D(:,i) = 0;
    end
end
clear i mi ma di
 
%%
X = D(:,1:end-1);
Y = D(:,end);
clear D
 
%% 
% hist(X(:,2),20)
% plot(Y,Y,'ob')
% hist(Y,20)
 
%%
n = size(X,1);
nl = round(.6*n);
nv = round(.2*n);
PD = pdist2(X,X);
 
i = randperm(n);
il = sort(i(1:nl));
iv = sort(i(nl+1:nl+nv));
it = sort(i(nl+nv+1:end));
err_best = Inf;
for gamma = logspace(-4,3,30)
    QLV = exp(-gamma*PD(il,il));
    QLT = exp(-gamma*PD([il,iv],[il,iv]));
    QV = exp(-gamma*PD(iv,il));
    QT = exp(-gamma*PD(it,[il,iv]));
    for lambda = logspace(-4,3,30)
        alpha = (QLV+lambda*eye(nl,nl))\Y(il);
        YP = QV*alpha;
        err_v = mean(abs(YP-Y(iv)));
        alpha = (QLT+lambda*eye(nl+nv,nl+nv))\Y([il,iv]);
        YP = QT*alpha;
        err_t = mean(abs(YP-Y(it)));
        fprintf('%e %e %e %e\n',gamma,lambda,err_v,err_t);
        if (err_v < err_best)
            err_best = err_v;
            err_model = err_t;
            gamma_best = gamma;
            lambda_best = lambda;
            YTrue = Y(it);
            YPred = YP;
        end
    end
end
figure, box on, grid on, box on
plot(YTrue,YPred,'ob')
title(sprintf('err = %f',err_model))