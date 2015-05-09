function [J grad] = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size*(input_layer_size+1)),hidden_layer_size,(input_layer_size+1));
Theta2 = reshape(nn_params((1+(hidden_layer_size*(input_layer_size+1))):end),num_labels,(hidden_layer_size+1));

m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

aux = zeros(num_labels,m);
for c = 1:num_labels
    aux(c,y==c) = 1;
end
y = aux;

A1 = [ones(1,size(X,1)); X'];
Z2 = Theta1 * A1;
A2 = [ones(1,size(X,1)); sigmoid(Z2)];
Z3 = Theta2 *  A2;
A3 = sigmoid(Z3);

J = 1/m*sum(sum(-y.*log(A3)-(1-y).*log(1-A3)));

J = J+lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

for t = 1:m
        
    d3 = A3(:,t) - y(:,t);
    d2 = (Theta2(:,2:end)'*d3).*sigmoidGradient(Z2(:,t));
    
    Theta1_grad = (Theta1_grad+d2*A1(:,t)');
    Theta2_grad = (Theta2_grad+d3*A2(:,t)');
    
end

Theta1_grad(:,1)     = 1/m  *Theta1_grad(:,1);
Theta1_grad(:,2:end) = 1/m  *Theta1_grad(:,2:end)   +lambda/m*Theta1(:,2:end);

Theta2_grad(:,1)     = 1/m  *Theta2_grad(:,1);
Theta2_grad(:,2:end) = 1/m  *Theta2_grad(:,2:end)   +lambda/m*Theta2(:,2:end);

grad = [Theta1_grad(:);Theta2_grad(:)];

end
