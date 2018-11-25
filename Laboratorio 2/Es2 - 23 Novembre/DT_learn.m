function T = DT_learn(X,Y,depth)
    % Stop criteria: the tree has height 0 or the data belong to the same class
    % In this case the current node is a leaf
    if (depth == 0 || length(unique(Y)) == 1)
       T.isleaf = true;
       T.class = mode(Y);                       % substitue with mean for regression problems (or the median if its distribution has multiple maximums)
    % Otherwise I have to check all the possible cuts and select the one
    % which minimizes the error. To find the cuts I have to sort the values
    % and check if two consecutive points have two different labels
    else
       T.isleaf = false;
       [n, d] = size(X);
       err_best = Inf;
       for i1 = 1:d
          [vX, i] = sort(X(:,i1), 'ascend');     % v is the sorted vector, i tells which was the position in the original vector
          vY = Y(i);
          for i2 = 1:n-1
             if (vY(i2) ~= vY(i2+1)) 
                 err = mean([vY(1:i2)~=mode(vY(1:i2)); ...
                             vY(i2+1:end)~=mode(vY(i2+1:end))]);
                 if (err_best > err)
                     err_best = err;
                     T.f = i1;                          % tree best feature
                     T.c = (vX(i2) + vX(i2+1)) / 2;     % tree best cut
                 end
             end
          end
       end
       f = X(:,T.f) <= T.c;
       T.left = DT_learn(X(f,:),Y(f),depth-1);
       T.right = DT_learn(X(~f,:),Y(~f),depth-1);
    end
end

