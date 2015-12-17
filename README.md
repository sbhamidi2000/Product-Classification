# Product-Classification
Otto Group Product Classification Challenge
The Otto Group is one of the world’s biggest e-commerce companies, with subsidiaries in more than 20 countries, including Crate & Barrel (USA), Otto.de (Germany) and 3 Suisses (France). We are selling millions of products worldwide every day, with several thousand products being added to our product line.

A consistent analysis of the performance of our products is crucial. However, due to our diverse global infrastructure, many identical products get classified differently. Therefore, the quality of our product analysis depends heavily on the ability to accurately cluster similar products. The better the classification, the more insights we can generate about our product range.

For this competition, we have provided a dataset with 93 features for more than 200,000 products. The objective is to build a predictive model which is able to distinguish between our main product categories. The winning models will be open sourced.

https://www.kaggle.com/c/otto-group-product-classification-challenge

Data
Each row corresponds to a single product. There are a total of 93 numerical features, which represent counts of different events. All features have been obfuscated and will not be defined any further.

There are nine categories for all products. Each target category represents one of our most important product categories (like fashion, electronics, etc.). The products for the training and testing sets are selected randomly.

File descriptions

trainData.csv - the training set
testData.csv - the test set
sampleSubmission.csv - a sample submission file in the correct format

Data fields
id - an anonymous id unique to a product
feat_1, feat_2, ..., feat_93 - the various features of a product
target - the class of a product

Approach
Idea is to come up with an ensemble classifier that performs well with new unseen test data. The features being masked it is difficult to decide each feature as categorical, numeric etc. Intial set of tests showed performance saturation with logistic, nueral net and SVD. Based on forum's collaboration focus was on two methods: xgboost (gradient boosted trees) and h20 deep learning (nueral network). Much of effort was incomplete by the deadline due to resource limitation, 32bit machine with 4Gb.

Result
Rank: 383/3514, Top 20%

