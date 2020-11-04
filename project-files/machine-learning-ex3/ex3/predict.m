function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%



% The feed-forward computation involves passing X to the hidden layer via Theta1,
% and subsequently the hidden layer to the output layer via Theta2.

% Add ones column to X
X = [ ones(m,1) X ];

% Compute hidden layer "a" (only 1 hidden layer in the given model)
a = sigmoid( X * Theta1' );

% Add ones column to hidden layer "a" for bias unit
a = [ ones(size(a,1),1) a];

% Compute output layer
h = sigmoid( a * Theta2' );

% o stores the max values of each row
% p stores the index of the max value, which corresponds to the class label from 1-9, 10 being "0".

[o, p] = max( h , [], 2);  %Returns a vector of the max of each row into a column vector






% =========================================================================


end
