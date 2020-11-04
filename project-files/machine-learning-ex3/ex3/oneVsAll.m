function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);   % In the given example, num_labels=10 and n=400

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%


for i = 1:num_labels
  %For each class, compute a classifier (i.e. theta vectors) corresponding to the ith class
  
  %Initial theta
  initial_theta = zeros(n + 1, 1);
  
  % Set options for fmincg
  options = optimset('GradObj', 'on', 'MaxIter', 50);   % TODO: Parameterize number of iterations if required
  
  %Run fmincg to obtain the optimal theta for the ith class
  %This is done by setting elements of y=1 only when an element equals i.
  %The optimal theta for this modified y will thus contain parameters that
  %correspond to positively classifying for "i", but not for other elements of y.
  [theta] = fmincg(@(t)(lrCostFunction(t, X, (y==i), lambda)), initial_theta, options);
  
  %Note that a cost can be returned as the second return value of fmincg (see code in fmincg.m)
  %Since this is not required, the cost matrix is not generated here.
  %TODO: The cost matrix may be added required.
  
  
  %Set the present optimal theta to the ith row of all_theta (row-wise required, so theta is transposed)
  all_theta(i,:) = theta';
  
  
endfor









% =========================================================================


end
