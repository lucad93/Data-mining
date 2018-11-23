function Y = DT_forw(T,X)
    n = size(X,1);
    Y = zeros(n, 1);
    for i = 1:n
        Ttmp = T;                               % "pointer" to the tree
        while (true)
           % Stopping criteria
           if (Ttmp.isleaf)
               Y(i) = Ttmp.class;
               break;
           % Otherwise i continue to explore the tree
           else
               if (X(i,Ttmp.f) <= Ttmp.c)
                   Ttmp = Ttmp.left;
               else
                   Ttmp = Ttmp.right;
               end
           end
        end
    end
end

