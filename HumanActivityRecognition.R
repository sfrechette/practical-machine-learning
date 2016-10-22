library(knitr)
opts_chunk$set(echo = TRUE, message=FALSE, results = 'hold')

library(caret)
library(randomForest)
library(corrplot)

set.seed(1535)

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "C:\\data\\pml-training.csv", method="curl")

## Warning: running command 'curl
## "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" -o
## "C:\data\pml-training.csv"' had status 127

## Warning in
## download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
## : download had nonzero exit status

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "C:\\data\\pml-testing.csv", method="curl")

## Warning: running command 'curl
## "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv" -o
## "C:\data\pml-testing.csv"' had status 127

## Warning in
## download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
## : download had nonzero exit status

trainingData = read.csv("C:\\data\\pml-training.csv", na.strings = c("NA", "#DIV/0!",""))

dim(trainingData)
## [1] 19622   160

testingData = read.csv("C:\\data\\pml-testing.csv", na.strings = c("NA", "#DIV/0!",""))

dim(testingData)
## [1]  20 160

trainingData<-trainingData[, -c(1:5)]
testingData<-testingData[, -c(1:5)]

naColumns <- colSums(is.na(trainingData)) > nrow(trainingData) * .5
trainingData <- trainingData[,!naColumns]

nearZeroVarColumns <- nearZeroVar(trainingData)
trainingData <- trainingData[, -nearZeroVarColumns]

testingData <- testingData[,!naColumns]
testingData <- testingData[, -nearZeroVarColumns]

inTrain <- createDataPartition(trainingData$classe, p=0.7, list=FALSE)
cleanTraining <- trainingData[inTrain,]
crossValidation <- trainingData[-inTrain,]

model = randomForest(classe ~., data=cleanTraining)
model

## 
## Call:
##  randomForest(formula = classe ~ ., data = cleanTraining) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.34%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3905    0    0    0    1 0.0002560164
## B    6 2649    3    0    0 0.0033860045
## C    0   14 2381    1    0 0.0062604341
## D    0    0   16 2236    0 0.0071047957
## E    0    0    0    6 2519 0.0023762376

prediction <- predict(model, crossValidation)
confMatrix <- confusionMatrix(prediction,crossValidation$classe)
outOfSampleError <- 1-sum(diag(confMatrix$table))/sum(confMatrix$table)
outOfSampleError
## [1] 0.002209006

print(confMatrix)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    1    0    0    0
##          B    0 1138    6    0    0
##          C    0    0 1020    5    0
##          D    0    0    0  959    1
##          E    0    0    0    0 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9978          
##                  95% CI : (0.9962, 0.9988)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9972          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9991   0.9942   0.9948   0.9991
## Specificity            0.9998   0.9987   0.9990   0.9998   1.0000
## Pos Pred Value         0.9994   0.9948   0.9951   0.9990   1.0000
## Neg Pred Value         1.0000   0.9998   0.9988   0.9990   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1934   0.1733   0.1630   0.1837
## Detection Prevalence   0.2846   0.1944   0.1742   0.1631   0.1837
## Balanced Accuracy      0.9999   0.9989   0.9966   0.9973   0.9995


testSetPrediction <- predict(model, testingData)
testSetPrediction
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E

pml_write_files = function(x){
n = length(x)
for(i in 1:n){
filename = paste0("problem_id_",i,".txt")
write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
}
}

pml_write_files(testSetPrediction)

