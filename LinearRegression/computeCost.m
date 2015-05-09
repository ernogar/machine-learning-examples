function [J, grad] = computeCost(X, y, theta, lambda)
%COMPUTECOST Compute cost and gradient for regularized linear regression with regularization
%   [J, grad] = COMPUTECOST(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y);
J = 0;
grad = zeros(size(theta));

J           = 1/(2*m)*  (X*theta-y)'*(X*theta-y)    +lambda/(2*m)*sum(theta(2:end).^2);
grad(1)     =     1/m*   X(:,    1)'*(X*theta-y);
grad(2:end) =     1/m*   X(:,2:end)'*(X*theta-y)    +lambda/m*theta(2:end);
grad = grad(:);

end
