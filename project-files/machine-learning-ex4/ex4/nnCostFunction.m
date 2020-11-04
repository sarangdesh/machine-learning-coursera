function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%








% Generalized number of output neurons, i.e. classes
K = max(y); % Here, 10 neurons, corresponding to numerals 1-9 and 0

% Add ones column to X to generate input layer
X = [ ones(m,1) X ];

%%%%% Compute forward propagation to get hypothesis %%%%%

% Compute activation of hidden layer (25 neurons in given example)
z2 = X * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];  % Add bias unit for hidden layer

% Compute activations for output layer (10 neurons in given example)
z3 = a2 * Theta2';
h = sigmoid(z3);  % Hypothesis 5000x10 for given example


% ONE-HOT ENCODING OF TRAINING EXAMPLE OUTPUTS
% Temporary matrix Y, corresponding to each training example output in the form of a vector
% If y(i) = 5, p is a vector [0;0;0;0;1;0;0;0;0;0]
% Values in y range from 1 to 10, with 10 --> "0"
% Y 5000x10 for given example
Y = zeros(size(y,1), K);
for t = 1 : size(y,1)       %For each training example
  Y( t, y(t,1) ) = 1;       % Update t-th column with the value of y in the t-th training example
endfor


% ALTERNATIVELY, use the bsxfun function with @eq argument for a one-line implementation
% Y = bsxfun(@eq, 1:K, y);  % Binary singleton function
% or...
% Y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);   % Repeat matrix function


% The cost function for logistic regression includes the term :
% [ y' * log(h) ) + (1-y) * log(1-h) ]

% In the present case though, y is transformed into Y, which requires an element-wise multiplication instead.

% FORM OF COST FUNCTION
% J = multiplier * [ costTerm ] + regularizationTerm

costTerm = [ Y .* log(h) ]  +  [ (1-Y) .* log(1-h) ];
% Note here that 5000x10 .* 5000x10 = 5000x10 for the given example

% Without regularization, J takes the form:
% J = (-1/m) * sum(costTerm);
% Here sum(costTerm) adds all elements of the resulting 5000x1 matrix 
% This version without regularization is not particularly useful, since
% regularization can be nullified by simply setting lambda=0.



%%%%%%%%%% GENERALIZED COST FUNCTION FOR SINGLE HIDDEN LAYER NEURAL NETWORK %%%%%%%%%%


% Removing bias units from input layer and hidden layer
Theta1NoBias = Theta1(:, 2:end);
Theta2NoBias = Theta2(:, 2:end);

% To calculate total cost, sum over all Y (each class and each training example), i.e. sum(costTerm)
% In each row of Y, only one non-zero element exists; so adding all elements yields the same answer.
J = (-1 / m) * sum(sum(costTerm)) + (lambda / (2 * m)) * (sum(sum(Theta1NoBias .^ 2)) + sum(sum(Theta2NoBias .^ 2)));



%%%%%%%%%% DELTA TERMS FOR SINGLE HIDDEN LAYER NEURAL NETWORK %%%%%%%%%%

delta_3 = h - Y;    % Difference between hypothesis and actual output from training data
delta_2 = (delta_3 * Theta2)(:, 2:end) .* sigmoidGradient(z2);

% Removing the bias unit for delta2:
% 5000x10 * 10x26 = 5000x26 --> 5000x25
% 5000x25 .* 5000x25 = 5000x25
% (for the given example)


%Partial derivatives of  J  w.r.t Thetas

Theta2_grad = (delta_3' * a2) / m;    % 10x5000 * 5000x26 = 10x26 for given example
%Adjusted to include regularization term without considering regularization for bias unit
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda * Theta2(:, 2:end) / m;    % 10x25 for given example

Theta1_grad = (delta_2' * X) / m;
%Adjusted to include regularization term without considering regularization for bias unit
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda * Theta1(:, 2:end) / m;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
