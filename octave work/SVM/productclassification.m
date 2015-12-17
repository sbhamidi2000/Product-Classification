%% Product Classification: SVM method

clear ; close all; clc

% Load Training Data
fprintf('Loading Data ...\n')

data = csvread('../../trainrun.csv');
%eval = csvread('../../testrun.csv');
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
y = myData(indices(1:train_indices), 95);

[X, mu, sigma] = featureNormalize(X);

Xval = myData(indices((train_indices+1):(train_indices+cv_indices)), 1:93);
yval = myData(indices((train_indices+1):(train_indices+cv_indices)), 95);

Xval = bsxfun(@minus, Xval, mu);
Xval = bsxfun(@rdivide, Xval, sigma);

fprintf('\nTraining Linear SVM (Product Classification)\n')
%Clear mem space
varlist = {'myData','data'};
clear(varlist{:})

C = 0.1;sigma = 0.1;

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 

p = svmPredict(model, Xval);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);


%% ================= Part 5: Top Predictors ====================

[weight, idx] = sort(model.w, 'descend');


fprintf('\nTop predictors: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', X{idx(i)}, weight(i));
end



%% =================== Test =====================

p = svmPredict(model, x);

