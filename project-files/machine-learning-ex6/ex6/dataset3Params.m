function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%




% Two alternative implementations are possible:
%     1. Useful for plotting and visualization:
%        Store all combinations of C_test and sigma_test and error corresponding to each combination.
%
%     2. Better performance:
%        Loop and store only minimum error value to return C and sigma.
%
% Here, the first type is implemented.



%%%% FIRST IMPLEMENTATION: With storage, useful for plotting, etc.

% Vary C between:      (0.01, 0.03, 0.1, 0,3, 1, 3, 10, 30)
% Vary sigma between:  (0.01, 0.03, 0.1, 0,3, 1, 3, 10, 30)
test_cases = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

gg = size(test_cases,1);

%Store all combinations of test cases for later retrieval
[p,q] = meshgrid(test_cases, test_cases);
parameter_list = [p(:) q(:)];

%Initialize error_list (for appendeding later to parameter_list)
error_list = zeros(gg*gg , 1);

i=1;

for p = 1:length(test_cases)    %Cycling through test cases for C
    
    for q = 1:length(test_cases)    %Cycling through test cases for sigma
        C_test = test_cases(p);
        sigma_test = test_cases(q);
        %Train model with X,y (training set)
        model= svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
        %Predict outputs for cross-validation set using trained model
        predictions = svmPredict(model, Xval);
        %Compute error against yval (validation set) for this model trained using current selection of C and sigma
        error = mean(double(predictions ~= yval));
        
        %Store error in a vector of size (gg*hh x 1) for later retrieval
        error_list(i) = error;
        i+=1;
    
    endfor
    
endfor


%Retrieve best C and sigma, i.e. where error is minimum
parameter_list = [parameter_list error_list];
[min_error min_id] = min(parameter_list(:,3))   %Get minimum error and id of minimum error
C = parameter_list(min_id,1);
sigma = parameter_list(min_id,2);




%%%% ALTERNATE IMPLEMENTATION: Better performance, no storage for later use

##minError = Inf;
##minC = Inf;
##minSigma = Inf;
##
##for p = 1:length(test_cases)
##    for q = 1:length(test_cases)
##        C_test = test_cases(p);
##        sigma_test = test_cases(q);
##        model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
##        predictions = svmPredict(model, Xval);
##        error = mean(double(predictions ~= yval));
##
##        if error < minError
##            minError = error;
##            minC = currentC;
##            minSigma = currentSigma;
##        end
##    end
##end
##
##C = minC;
##sigma = minSigma;



% =========================================================================

end
