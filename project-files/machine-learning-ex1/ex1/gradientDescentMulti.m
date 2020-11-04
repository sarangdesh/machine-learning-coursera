function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
featureCount = size(X,2); %Number of features
printf('Feature count = %0.0f \n\n', featureCount);
Delta = zeros(featureCount,1);  %column vector

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    % theta = theta - alpha * Delta
    % where Delta = partial differential of J(theta)
    
    for f = 1:featureCount
      
      Delta(f,1) = (1/m) * sum( ((theta' * X') - y') * X(:,f) );
      theta = theta - alpha*Delta;
      
    endfor



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

disp(J_history);

end
