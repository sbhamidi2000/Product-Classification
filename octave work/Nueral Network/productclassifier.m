
%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 93;  % 93 features
hidden_layer_size = 60;   % 50 hidden units
num_labels = 9;          % 9 labels, from 1 to 9   
                          

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading Data ...\n')

data = csvread('../../trainrun.csv');
%eval = csvread('../../testrun.csv');
%eval = eval(2:end,1:38);


myData = data(2:end,2:104);
 myData(:,94) = [];

indices = randperm(length(myData));
train_indices = round(length(myData)*0.7);
cv_indices = round(length(myData)*0.15);
test_indices = round(length(myData)*0.15); %prevent rounding overrun
endpoint = train_indices + cv_indices + test_indices;

if(endpoint> length(myData)) endpoint= endpoint-1;
endif

X = myData(indices(1:train_indices), 1:93);
y = myData(indices(1:train_indices), 94:102);

[X, mu, sigma] = featureNormalize(X);

Xval = myData(indices((train_indices+1):(train_indices+cv_indices)), 1:93);
yval = myData(indices((train_indices+1):(train_indices+cv_indices)), 94:102);

Xval = bsxfun(@minus, Xval, mu);
Xval = bsxfun(@rdivide, Xval, sigma);
 
 
 m = size(X, 1);




%% ================ Part 2: Loading Parameters ================
% In this part of the exercise, we load some pre-initialized 
% neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
init_epsilon = 0.0001;
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta3 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta4 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:) ; ...
                     initial_Theta4(:)];
nn_params= initial_nn_params;

%% ================ Part 3: Compute Cost (Feedforward) ================
%  To the neural network, you should first start by implementing the
%  feedforward part of the neural network that returns the cost only. You
%  should complete the code in nnCostFunction.m to return cost. After
%  implementing the feedforward to compute the cost, you can verify that
%  your implementation is correct by verifying that you get the same cost
%  as us for the fixed debugging parameters.
%
%  We suggest implementing the feedforward cost *without* regularization
%  first so that it will be easier for you to debug. Later, in part 4, you
%  will get to implement the regularized cost.
%
%fprintf('\nFeedforward Using Neural Network ...\n')

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);



%% =============== Part 4: Implement Regularization ===============
%  Once your cost function implementation is correct, you should now
%  continue to implement the regularization with the cost.
%

%fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% Weight regularization parameter (we set this to 1 here).
%lambda = 1;

%J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
%                   num_labels, X, y, lambda);



%% ================ Part 5: Sigmoid Gradient  ================
%  Before you start implementing the neural network, you will first
%  implement the gradient for the sigmoid function. You should complete the
%  code in the sigmoidGradient.m file.
%

fprintf('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient([1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');


%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta3 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta4 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:) ; initial_Theta3(:) ; ...
                     initial_Theta4(:)];

%% =============== Part 7: Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;



%% =============== Part 8: Implement Regularization ===============
%  Once your backpropagation implementation is correct, you should now
%  continue to implement the regularization with the cost and gradient.
%

%fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
%lambda = 3;
%checkNNGradients(lambda);

% Also output the costFunction debugging values
%debug_J  = nnCostFunction(nn_params, input_layer_size, ...
 %                         hidden_layer_size, num_labels, X, y, lambda);

%fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f ' ...
 %        '\n(this value should be about 0.576051)\n\n'], debug_J);



%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 1000); %fmincg
%options = optimset('GradObj', 'on', 'MaxIter', 400); %fminunc

%  You should also try different values of lambda

%lambdavec= [0 0.1 0.2 0.3 0.4 0.5 0.8 1.0 5.0 10.0]
lambdavec= [0.5]
for i= 1:length(lambdavec)
lambda = lambdavec(i);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)

%[nn_params, cost] = fminunc(costFunction, initial_nn_params, options);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
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



%% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

%fprintf('\nVisualizing Neural Network... \n')

%displayData(Theta1(:, 2:end));



%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, Theta3, Theta4, Xval);
 [dummy, ysel] = max(yval, [], 2);
fprintf('\nLambda= %f  Training Set Accuracy: %f\n', lambda, mean(double(pred == ysel)) * 100);

end
