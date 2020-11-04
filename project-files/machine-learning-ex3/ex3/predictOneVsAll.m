function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       




% SIMPLIFIED EXAMPLE
% Suppose the following:
% all_theta = 10x3 (10 classes, 3 features)
% X = 50x3  (50 examples, 3 features)
% X * all_theta' = 50x10 (predictions for each example corresponding to each class)

% To offer a classification response for one example, need to choose the highest
% "probability" among the elements of the example's row in (X * all_theta').

% Each element in p corresponds to the max value of each row in (X * all_theta').
% Therefore:


temp = sigmoid( X * all_theta' );


% o stores the max values of each row
% p stores the index of the max value, which corresponds to the class label from 1-9, 10 being "0".

[o, p] = max( temp , [], 2);  %Returns a vector of the max of each row into a column vector






% =========================================================================


end
