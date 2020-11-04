function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
featureCount = size(X,2); %Number of features
Delta = zeros(featureCount,1);  %column vector

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % theta = theta - alpha * Delta
    % where Delta = partial differential of J(theta)
     
    % For univariate, partial differential w.r.t thetaZero and thetaOne respectively is known.
    % But this does not help generalize the model for any number of features
    
    DeltaZero = (1/m) * sum( (theta' * X') - y' );
    DeltaOne =  (1/m) * sum( ((theta' * X') - y') * X(:,2) );
    
    thetaZero = theta(1,1) - alpha*DeltaZero;   %Update thetaZero
    thetaOne = theta(2,1) - alpha*DeltaOne;     %Update thetaOne
    
    theta(1,1) = thetaZero;
    theta(2,1) = thetaOne;
    
    % Update theta (generalized for multivariate)
    % for p = 1:length(theta)
    %   theta(1,p) = thetaZero;
    %   theta(2,p) = thetaOne;  
    % endfor
    
    
    %%% GENERAL METHOD FOR ANY NUMBER OF FEATURES
    %% Note that this method doesn't get recognized in the submit script for grading the assignment.
    %for f = 1:featureCount
      
    %  Delta(f,1) = (1/m) * sum( ((theta' * X') - y') * X(:,f) );
    %  theta = theta - alpha*Delta;
      
    %endfor
    
    
    % ============================================================

    % Save the cost J in every iteration for updated theta
    J_history(iter) = computeCost(X, y, theta);

end

end
