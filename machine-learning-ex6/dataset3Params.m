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
candidate = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
prediction_result = zeros((size(candidate, 1) ^ 2), 3);
length_of_result = size(prediction_result, 1);
cursor = 1;

for i = 1:size(candidate, 1)
  C = candidate(i, :);
  for j = 1:size(candidate, 1)
    sigma = candidate(j, :);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    prediction = svmPredict(model, Xval);
    prediction_error = mean(double(prediction ~= yval));
    prediction_result(cursor, :) = [C sigma prediction_error];
    cursor += 1;
  end
end
% prediction_result(:, :)
[minus_values, row_index] = min(prediction_result);
C = prediction_result(row_index(3), 1);
sigma = prediction_result(row_index(3), 2);


% =========================================================================

end
