function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

J = [ -(1/m) * [(y'*log(sigmoid(X*theta))) + (1-y)'*log(1-sigmoid(X*theta))] ]  +  [ (lambda/(2*m)) * [sum(theta.^2)-theta(1)^2] ] ;
% The Theta0=theta(1) term is not to be regularized, hence is subtracted from the summation.

temp = (1/m) * ( ( sigmoid(X*theta) - y )' * X )';
grad = temp + (lambda/m)*theta;
% 3x1

% After calculating grad for all terms with regularization formula,
% need to replace grad term for j=0, i.e. grad(1) with its formula.
grad(1) = temp(1);





% =============================================================

end
