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

## init test_vec and error rate of predictions for every combination of test values
test_vec = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]';
err_pred = zeros(length(test_vec), length(test_vec));

for i = 1:length(test_vec)
    for j = 1:length(test_vec)
        % SVM Parameters
        C = test_vec(i); sigma = test_vec(j);
        
        % SVM train 
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        
        % SVM predict
        pred = svmPredict(model, Xval);
        
        % SVM predict error
        err_pred(i,j) = mean(double(pred ~= yval))
    endfor
endfor

## find out the index of the minimal error
[C_index,sigma_index] = find(err_pred == min(min(err_pred)))

C = test_vec(C_index);
sigma = test_vec(sigma_index);


% =========================================================================

end
