function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X =     reshape(params(1:num_movies*num_features)   ,num_movies,num_features);
Theta = reshape(params(num_movies*num_features+1:end),num_users,num_features);
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

J     = (1/2)    *sum(sum(R.*((X*Theta'-Y).^2)))    +(lambda/2)*sum(sum(Theta.^2))     +(lambda/2)*sum(sum(X.^2));

for i = 1:num_movies
    idx = find(R(i, :)==1);
    X_grad(i,:)     = (X(i,:)*Theta(idx,:)'-Y(i,idx))*Theta(idx,:)    +lambda*X(i,:);
end
for i = 1:num_users
    idx = find(R(:, i)==1);
    Theta_grad(i,:) = (X(idx,:)*Theta(i,:)'-Y(idx,i))'*X(idx,:)       +lambda*Theta(i,:);
end

grad = [X_grad(:);Theta_grad(:)];

end
