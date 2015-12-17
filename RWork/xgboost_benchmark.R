###xgboost could not be installed###

require(xgboost)
require(methods)

train = read.csv('../train.csv',header=TRUE,stringsAsFactors = F)
test = read.csv('../test.csv',header=TRUE,stringsAsFactors = F)
train = train[,-1]
test = test[,-1]

#Shuffle
trainshuffle <- train[sample(nrow(train)),]
train <- trainshuffle

#Scale
# for(i in 1:93){
#   train[,i] <- as.numeric(train[,i])
#   train[,i] <- sqrt(train[,i]+(3/8))
#   test[,i] <- as.numeric(test[,i])
#   test[,i] <- sqrt(test[,i]+(3/8))
# }

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

#x = rbind(train[,-ncol(train)],test)
x = train[,-ncol(train)]

x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teindr = nrow(test)

test = as.matrix(test)
test = matrix(as.numeric(test),nrow(test),ncol(test))

predsum <- read.csv("../sampleSubmission.csv")
predsum[,2:10] <- 0


max.depth=c(13)
eta= c(0.3)
min_child_weight= c(3)
colsample_bytree= c(0.6)
row_subsample= c(0.8)
min_loss_reduction= c(1)
column_subsample= c(1.0)

for(i in 1:2) {
# Set necessary parameter

 cat("iteration = ", i, "\n") 
  
  
  param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8,
#               "max.depth"=sample(max.depth,1), 
#               "eta"=sample(eta,1),"min_child_weight"=sample(min_child_weight,1),
#               "colsample_bytree"= sample(colsample_bytree,1),"row_subsample"=sample(row_subsample,1) ,
#               "min_loss_reduction"=sample(min_loss_reduction,1), 
#               "column_subsample"=sample(column_subsample,1))
"max.depth"=12, 
"eta"=0.05,"min_child_weight"=2,
"colsample_bytree"= 0.8,"row_subsample"=0.8 ,
"min_loss_reduction"=1, 
"column_subsample"=1)

  print(param, row.names = FALSE)

# Run Cross Validation
cv.nround = 250
bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                nfold = 10, nrounds=cv.nround)

# Train the model
nround = 150
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

# Make prediction
pred = predict(bst,test)
 pred = matrix(pred,9,length(pred)/9)
pred = t(pred)
# 
# # Output submission
# pred = format(pred, digits=2,scientific=F) # shrink the size of submission
 pred = data.frame(1:nrow(pred),pred)
predsum[,2:10] <- predsum[,2:10] + pred[,2:10]

print(i)
}
names(predsum) = c('id', paste0('Class_',1:9))
write.csv(predsum,file='submission.csv', quote=FALSE,row.names=FALSE)
