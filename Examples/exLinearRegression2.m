%  Exercise 1: Linear regression with multiple variables

%% Clear and Close Figures
clear ; close all; clc; restoredefaultpath;
addpath('../LinearRegression');
addpath('../Utils');
addpath('Other/LinReg');
%% ================ Feature Normalization ================

fprintf('Loading data ...\n');

%% Load Data
data = load('Data/exLinearRegression2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.1;
num_iters = 50;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

hold on;

alpha = 0.3;
theta = zeros(3, 1);
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);
plot(1:numel(J_history), J_history, '-r', 'LineWidth', 2);

alpha = 0.5;
theta = zeros(3, 1);
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);
plot(1:numel(J_history), J_history, '-g', 'LineWidth', 2);

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
price = 0; % You should change this
new = [1 ([1650 3]-mu)./sigma];
price = new*theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Normal Equations ================

fprintf('Solving with normal equations...\n');

%% Load Data
data = csvread('Data/exLinearRegression2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
price = [1 1650 3]*theta; % You should change this

fprintf(['Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f\n'], price);

restoredefaultpath;