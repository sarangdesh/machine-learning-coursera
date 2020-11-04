function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% For the given example:
% y       nx1
% X       nx3 (2 features + Theta0)
% theta   3x1




% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% J(Theta) = -(1/m) ?[ ylog( h(x) ) + (1-y)log( 1 - h(x) ) ]

J = -(1/m) * [(y'*log(sigmoid(X*theta))) + (1-y)'*log(1-sigmoid(X*theta))];
% [1xn * (nx3 * 3x1)] - [1xn * (nx3 * 3x1)]
% =[1x1]-[1x1]
% = 1x1
% No need to use the sum function as required in linear regression cost function.

grad = (1/m) * ( ( sigmoid(X*theta) - y )' * X )';
% [(nx3 * 3x1) - nx1]' * nx3
% = (1xn * n*3)'
% = 3x1



% =============================================================

end
