function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;   %X = [houseSize bedroomCount] --> mx2 matrix
mu = zeros(1, size(X, 2));      %mu = [mu1, mu2]
sigma = zeros(1, size(X, 2));   %sigma = [sigma1, sigma2]

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

m = size(X,1);  %Number of examples
featureCount = size(X,2);   %Number of features


for f = 1:featureCount
  
  % Compute mu and sigma for all features
  mu(1,f) = mean(X(:,f));
  sigma(1,f) = std(X(:,f));
  
  % Normalize data for all features
  X_norm(:,f) = ( X(:,f) - mu(1,f) ) / sigma(1,f);
  
endfor

disp(mu);
disp(sigma);


% ============================================================

end
