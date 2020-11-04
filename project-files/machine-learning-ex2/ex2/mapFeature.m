function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

degree = 6;
out = ones(size(X1(:,1)));    %Since X1 and X2 can only be the same size,
                              %this equation might as well contain X2.
                              
                              % out is a size(X1) x 1 matrix full of ones
                              
for i = 1:degree
    for j = 0:i
      
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
        
        % Sarang's note:
        % In index expressions the keyword 'end' automatically refers to the last
        % entry for a particular dimension. This magic index can also be used
        % in ranges and typically eliminates the needs to call size or length
        % to gather array bounds before indexing.
        
        % Here, it serves to add a new column in each iteration.
        % This way, the first column remains full of ones to correspond to Theta1.
    end
end

end