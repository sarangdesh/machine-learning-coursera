function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

sim = exp ( -(norm(x1-x2))^2 / (2 * sigma^2));

%The norm function is essentially a p-norm function, i.e. (sum(x(i)^p))^(1/p).
%If p is not specified, p=2 by default for getting the Euclidean/Frobenius norm.


% Alternatively:
%sim = exp(-1*sum(abs(x1-x2).^2)/(2*sigma^2));
%sim = exp(-(x1 - x2)' * (x1 - x2) / (2 * sigma * sigma));
%sim = exp(-(sum((x1-x2).^2))/(2*(sigma^2)));



% =============================================================
    
end