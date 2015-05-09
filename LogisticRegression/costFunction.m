function [J, grad] = costFunction(theta, X, y, lambda)
%COSTFUNCTION Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);
J = 0;
grad = zeros(size(theta));

J           = 1/m*  sum(-y.*log(sigmoid(X*theta))-(1-y).*log(1-sigmoid(X*theta)))   +lambda/(2*m)*sum(theta(2:end).^2);
grad(1)     = 1/m* X(:,    1)'*(sigmoid(X*theta)-y);
grad(2:end) = 1/m* X(:,2:end)'*(sigmoid(X*theta)-y)                                 +lambda/m*theta(2:end);
grad = grad(:);

end
