function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

## 1. feedforward propagation and cost function

for i = 1:m
    ## feedforward propagation
    a1 = X(i,:);            % a1        1 X 400
    a1_bias = [1, a1]';     % a1_bias   401 X 1
    z2 = Theta1*a1_bias;    % Theta1    25 X 401
    a2 = sigmoid(z2);       % a2        25 X 1
    a2_bias = [1; a2];      % a2_bias   26 X 1
    z3 = Theta2*a2_bias;    % Theta2    10 X 26
    a3 = sigmoid(z3);       % a3        10 X 1
    
    ## label in vector form
    y_vec_i = zeros(num_labels,1);
    y_vec_i(y(i)) = 1;      % y_vec_i   10 X 1: [0 0 0 0 1 0 0 0 0 0]'
    
    ## cost function   
    for k = 1:num_labels
        J = J + (-y_vec_i(k))*log(a3(k)) - (1-y_vec_i(k))*log(1-(a3(k)));
    endfor

endfor

J = J/m;

## 2. back propagation

for t = 1:m
    ## step 1: feedforward propagation
    a1 = X(t,:);            % a1        1 X 400
    a1_bias = [1, a1]';     % a1_bias   401 X 1
    z2 = Theta1*a1_bias;    % Theta1    25 X 401
    a2 = sigmoid(z2);       % a2        25 X 1
    a2_bias = [1; a2];      % a2_bias   26 X 1
    z3 = Theta2*a2_bias;    % Theta2    10 X 26
    a3 = sigmoid(z3);       % a3        10 X 1
    
    ## step 2: delta in layer 3 
    ## label in vector form
    y_vec_t = zeros(num_labels,1);
    y_vec_t(y(t)) = 1;      % y_vec_i   10 X 1: [0 0 0 0 1 0 0 0 0 0]'
    
    delta3 = a3 - y_vec_t;  % delta3 10 X 1
    
    ## step 3: delta in layer 2
    delta2 = Theta2(:,2:end)'*delta3.*sigmoidGradient(z2);   % delta2 25 X 1
    
    ## step 4: Accumulate the gradient from this example t
    Theta1_grad = Theta1_grad + delta2*(a1_bias');    % Delta1 25 X 400
    Theta2_grad = Theta2_grad + delta3*(a2_bias');   % Delta2 10 X 26
endfor
## step5: gradient
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;


## 3. regularized cost function

R = 0;
for i = 1:hidden_layer_size
    for j = 2:(input_layer_size+1)
        R = R + Theta1(i,j)*Theta1(i,j);
    endfor
endfor

for i = 1:num_labels
    for j = 2:(hidden_layer_size+1)
        R = R + Theta2(i,j)*Theta2(i,j);
    endfor
endfor
R = lambda/(2*m)*R;

J = J + R;


## 4. regularized gradient
for i = 1:hidden_layer_size
    for j = 2:(input_layer_size+1)
        Theta1_grad(i,j) = Theta1_grad(i,j) + Theta1(i,j)*lambda/m;
    endfor
endfor

for i = 1:num_labels
    for j = 2:(hidden_layer_size+1)
        Theta2_grad(i,j) = Theta2_grad(i,j) + Theta2(i,j)*lambda/m;
    endfor
endfor








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
