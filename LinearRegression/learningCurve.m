function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = LEARNINGCURVE(X, y, Xval, yval, lambda) 
%       returns the train and cross validation set errors for a learning
%       curve. In particular, it returns two vectors of the same length
%       error_train and error_val.
%       Then, error_train(i) contains the training error for i examples
%       (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.

m = size(X, 1);
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m

    steps = 10;
    for step = 1:steps
    
        rndIDX = randperm(m); 
        newX = X(rndIDX(1:i),:);
        newy = y(rndIDX(1:i));
        rndIDX = randperm(size(Xval, 1)); 
        newXval = Xval(rndIDX(1:i), :);
        newyval = yval(rndIDX(1:i));

        [theta] = trainLinearReg(newX, newy, lambda);
        [rand_train, dump] = computeCost( newX, newy, theta, 0);
        [rand_val, dump] = computeCost( newXval, newyval, theta, 0);
        error_train(i) =  error_train(i) + rand_train;
        error_val(i) =  error_val(i) + rand_val;
        
    end
    error_train(i) = error_train(i) / steps;
    error_val(i) = error_val(i) / steps;
    
end

end
