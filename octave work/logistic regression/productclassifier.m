%% Otto product classifier | Part 1: One-vs-all



%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 93;  % 93 features
num_labels = 9;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% Load Data
data = csvread('../../trainrun.csv');


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

X_poly = polyMolyFeatures(X, 2);
[X_poly, mu, sigma] = featureNormalize(X_poly);

Xval = myData(indices((train_indices+1):(train_indices+cv_indices)), 1:93);
yval = myData(indices((train_indices+1):(train_indices+cv_indices)), 94:102);
Xval_poly = polyMolyFeatures(Xval, 2);

Xval_poly = bsxfun(@minus, Xval_poly, mu);
Xval_poly = bsxfun(@rdivide, Xval_poly, sigma);

%Xtest = myData(indices((train_indices+cv_indices+1):endpoint), 1:93);
%ytest = myData(indices((train_indices+cv_indices+1):endpoint), 94:102);
%Xtest_poly = polyMolyFeatures(Xtest, 2);
%Xtest_poly = bsxfun(@minus, Xtest, mu);
%Xtest_poly = bsxfun(@rdivide, Xtest, sigma);

%Clear mem space
%varlist = {'myData','data','X','Xval'};
%clear(varlist{:})



m = size(X, 1);



%% ============ Part 2: Vectorize Logistic Regression ============

fprintf('\nTraining One-vs-All Logistic Regression...\n')
%lambda_vec = [0,0.1,0.2,0.4,0.8,1.0,10,100];
lambda_vec = [0.1];

lambda_fix = []
for i = 1:length(lambda_vec)
lambda = lambda_vec(i);

[all_theta] = oneVsAll(X_poly, y, num_labels, lambda);


%% ================ Part 3: Predict for One-Vs-All ================
%  After ...
pred = predictOneVsAll(all_theta, Xval_poly);
[max_y ysel] = max(yval, [],2);

fprintf('\nTraining Set Accuracy: %f\n', (accuracy = mean(double(pred == ysel)) * 100));

lambda_fix = [lambda_fix lambda accuracy];

end
result = [ysel pred]
result1 = result;
[i,j] = find(result1(:,2)!=result1(:,1));
result1 =  result1(i,:);

u=unique(result1(:,1));

for i=1:9
res=arrayfun(@(y)length(result1(result1(:,1)==y & result1(:,2)==i)),u);
fprintf ('Count %d\n',i);
fprintf ('%d\n',res);
end