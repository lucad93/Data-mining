function [XT,YP,best_error,best_lambda,best_gamma] = KRLS(X,Y,lset_dim)
    if (lset_dim < 0 || lset_dim > 1)
       error('Learning set dimension must be a valid percentage'); 
    end
    n = size(X,1); % number of total samples
    d = size(X,2); % number of features
    nl = round(lset_dim * n); % number of learning set samples
    PD = pdist2(X,X); % pair-wise distance: euclidean distance between samples
    err = zeros(30*30,1); % array of errors, for every possible value of lambda and gamma
    
    % Learning phase
    % 30 iterations for every value of lambda and gamma
    for k = 1:30
        % Construction of learning and test set
        i = randperm(n); % random permutation of indexes
        il = sort(i(1:nl)); % array of learning set indexes (sorted for optimisation)
        it = sort(i(nl+1:end)); % array of test set indexes
        j = 0;
        % Loop through 30 different lambda and gamma values, in a logarithmic space between 10^-4 and 10^3
        for gamma = logspace(-4,3,30)
            QL = exp(-gamma * PD(il, il));
            QT = exp(-gamma * PD(it, il));
            for lambda = logspace(-4,3,30)
                j = j + 1;
                alpha = (QL + lambda * eye(nl,nl))\Y(il); % contribute (weight) of the current sample           
                YP = QT * alpha; % prediction attempt using the current model
                err(j) = err(j) + mean(abs(YP-Y(it))) / 30; % computation of the average error of 30 iterations
            end
        end
    end
    
    % Research of best error, lambda and gamma
    j = 0;
    best_error = Inf;
    for gamma = logspace(-4,3,30)
       for lambda = logspace(-4,3,30)
          j = j + 1;
          if (err(j) < best_error)
              best_error = err(j);
              best_gamma = gamma;
              best_lambda = lambda;
          end
       end
    end
    
    % Creation of a linear space for every feature
    m = 1000; % sampling rate of the linear space
    XT = zeros(m,d);
    for k = 1:d
        XT(:,k) = linspace(min(X(:,k)), max(X(:,k)), m);
    end
    
    % Computation of the prediction on XT, using the best parameters
    Q = exp(-best_gamma * PD);
    alpha = (Q + best_lambda * eye(n,n)) \ Y;
    YP = exp(-best_gamma * pdist2(XT,X)) * alpha;
end