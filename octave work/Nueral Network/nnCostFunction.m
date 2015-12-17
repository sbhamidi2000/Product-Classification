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
range1e = (hidden_layer_size * (input_layer_size + 1));
range2e = (hidden_layer_size *(hidden_layer_size+1) +range1e);
range3e = (hidden_layer_size *(hidden_layer_size+1) +range2e);

Theta1 = reshape(nn_params(1:range1e), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + range1e):range2e), ...
                 hidden_layer_size, (hidden_layer_size + 1));
                 
Theta3 = reshape(nn_params((1 + range2e):range3e), ...
                 hidden_layer_size, (hidden_layer_size + 1));

Theta4 = reshape(nn_params((1 + range3e):end), ...
                 num_labels, (hidden_layer_size + 1));
               
% Setup some useful variables
m = size(X, 1);
X = [ones(m, 1) X]; %add bias column to input features
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));
Theta4_grad = zeros(size(Theta4));

%transform y
%yTransf = eye(num_labels);
%yVec = yTransf(y, :);
yVec=y;
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

a1 = X;

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);
a3 = [ones(size(a3,1), 1) a3];

z4 = a3 * Theta3';
a4 = sigmoid(z4);
a4 = [ones(size(a4,1), 1) a4];

z5 = a4 * Theta4';
a5 = sigmoid(z5);

%Remove first column from both Theta matrices for regularization
Theta1p = Theta1(:,[2:size(Theta1,2)]);
Theta2p = Theta2(:,[2:size(Theta2,2)]);
Theta3p = Theta3(:,[2:size(Theta3,2)]);
Theta4p = Theta4(:,[2:size(Theta4,2)]);


J = sum(sum((-log(a5).*yVec)-(log(1-a5).*(1-yVec))))*(1/m) + ...
                ((0.5*lambda)/m)* ...% Regularization part
                (sum(Theta4p(1:end).^2)+ ...
                 sum(Theta3p(1:end).^2)+ sum(Theta2p(1:end).^2)+ sum(Theta1p(1:end).^2)); 
                
%fprintf (['Theta1 = %f \n'], sum(Theta1(2:end).^2));


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

d5 = a5- yVec;
d4 = (d5 * Theta4p).* sigmoidGradient(z4);
d3 = (d4 * Theta3p).* sigmoidGradient(z3);
d2 = (d3 * Theta2p).* sigmoidGradient(z2);


Delta1 = d2' *a1;
Delta2 = d3' *a2;
Delta3 = d4' *a3;
Delta4 = d5' *a4;



Theta1_grad = (1/m) *Delta1;
Theta2_grad = (1/m) *Delta2;
Theta3_grad = (1/m) *Delta3;
Theta4_grad = (1/m) *Delta4;




%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta3(:,1) = 0;
Theta4(:,1) = 0;



Theta1_grad = Theta1_grad + (lambda/m)*Theta1;
Theta2_grad = Theta2_grad + (lambda/m)*Theta2;
Theta3_grad = Theta3_grad + (lambda/m)*Theta3;
Theta4_grad = Theta4_grad + (lambda/m)*Theta4;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:); Theta4_grad(:)];


end
