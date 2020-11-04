function randomPlotting(X_poly, y, X_poly_val, yval, lambda, loops)
  
  
%% =========== Optional Exercise: Plotting learning curves with randomly selected examples =============


% Note that within the given dataset, X is 12x1, X_poly is 12x9, and Xval is 21x1.


m = size(X_poly, 1);

% vectors for storing training errors and cross validation errors
error_train = zeros(m,1);
error_val =  zeros(m,1);

lambda = 0.01;  % to compare to given solution
% number of times to loop for picking random values


% pick a random value "loops" times, summarise the calculated the errors, to below doing an average
	for l=1:loops
    
		for j=1:m
			
      %Randomly generate for training set
      seq = randperm(m,j);%produce a row vector of random integers j in [1,m]
      % 'seq' serves as the index for randomly selected corresponding elements in X_poly and y
      X_poly_rand = X_poly(seq,:);
      y_rand = y(seq,:);
      
      %Randomly generate for cross validation set
      seq_val = randperm(m,j);
      X_poly_val_rand = X_poly_val(seq_val,:);
      yval_val_rand = yval(seq_val,:);
      
      [theta] = trainLinearReg(X_poly_rand, y_rand, lambda);
      lam = 0; %lamda=0 for error computation
      [J, grad] = linearRegCostFunction(X_poly_rand, y_rand, theta, lam);
      [Jval, gradval] = linearRegCostFunction(X_poly_val_rand, yval_val_rand, theta, lam);
      
      %Maintain sum of errors
      error_train(j) = error_train(j) + J;
      error_val(j) = error_val(j) + Jval;

    endfor
  
	endfor

	% finding the average
	error_train = error_train ./ loops;
	error_val = error_val ./ loops;

	% least but not last, do some plotting to visualise our results
	plot(1:m, error_train, 1:m, error_val);
	xlabel('Number of training examples');
	ylabel('Error');
	axis([0 13 0 100]);
	legend('Train', 'Cross Validation');




























% Randomly select i examples from the training set and cross-validation set

% Using the polynomial training set generated,
% learn the parameters from the randomly chosen trainig set
% and evaluate the parameters on the randomly chosen cross-validation set.
% Perform the operation many times and average across each.

%repeat_count = 50;

%for k = 1:repeat_count
  
  % Choose a random value from X_poly, and store the corresponding y value as well.
  
  
  %X_poly_random = randsample(X_poly, random_count);
  %y_random = randsample(y, random_count);
  %[theta] = trainLinearReg(X_poly_random, y, lambda);

  %[error_train, error_val] = learningCurve(X_poly_random, y, X_poly_val, yval, lambda);
  % plot(1:m, error_train, 1:m, error_val);
  
%endfor

%error_train_average = mean(error_train);
%fprintf('Average training error for %f repeats and %f random samples = %f)\n\n', repeat_count, random_count, error_train_average);







  
  
endfunction
