function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

## Cost
J = sum((X*theta-y).^2)/(2*m);              % cost
Reg = sum(theta(2:end).^2)*lambda/(2*m);    % regularization
J = J + Reg;                                % cost with regularization     

## Gradient
grad(1) = sum((X*theta-y).*X(:,1))/m;
for i = 2:length(theta)
    grad(i) = sum((X*theta-y).*X(:,i))/m + theta(i)*lambda/m; 
endfor






% =========================================================================

grad = grad(:);

end