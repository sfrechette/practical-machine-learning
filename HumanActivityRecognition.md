---
title: "Human Activity Recognition - Project | Practical Machine Learning"
author: "sfrechette"
date: "January 23, 2015"
output: html_document
---

##Goal of the project
In this project, we're given a training set at
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>
and a test set at <https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv> from the source <http://groupware.les.inf.puc-rio.br/har>
These sets contains data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants who are asked to perform barbell lifts correctly and incorrectly in 5 different ways. Our goal is to predict the manner in which the people did the exercise, i.e. the "classe" variable in the training set. After training and creating a model, we will use this model to predict 20 different test cases in the test data set. We will submit our final predictions to Coursera and see how well we did :)

##Preparation for R Markdown

```r
library(knitr)
opts_chunk$set(echo = TRUE, message=FALSE, results = 'hold')
```


##Loading all the needed libraries first

```r
library(caret)
library(randomForest)
library(corrplot)
```


##Setting the seed

```r
set.seed(1535)
```


##Downloading testing and training data

Let's download the 2 files from the internet first.

```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "C:\\data\\pml-training.csv", method="curl")
```

```
## Warning: running command 'curl
## "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv" -o
## "C:\data\pml-training.csv"' had status 127
```

```
## Warning in
## download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
## : download had nonzero exit status
```

```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "C:\\data\\pml-testing.csv", method="curl")
```

```
## Warning: running command 'curl
## "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv" -o
## "C:\data\pml-testing.csv"' had status 127
```

```
## Warning in
## download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
## : download had nonzero exit status
```

Now we should get what's inside the downloaded training .csv file. When reading the files, we'll also specify a vector of values that should be treated as NAs.

```r
trainingData = read.csv("C:\\data\\pml-training.csv", na.strings = c("NA", "#DIV/0!",""))
dim(trainingData)
```

```
## [1] 19622   160
```
We see that the training data has 19622 observations of 160 variables.

Let's do the same for our testing .csv file.

```r
testingData = read.csv("C:\\data\\pml-testing.csv", na.strings = c("NA", "#DIV/0!",""))
dim(testingData)
```

```
## [1]  20 160
```
We see that the testing data has 20 observations of 160 variables.

#Reducing the number of predictors
Looking at the predictors, logically the first 5 of them (id, user name and timestamps) shouldn't have anything to do with our model. So let's remove them from both the training and testing data sets.

```r
trainingData<-trainingData[, -c(1:5)]
testingData<-testingData[, -c(1:5)]
```
Now, we're left with 155 columns.

Another thing that draws our attention in the data is that there are many NA values. Let's remove columns which have more than 50% of "NA".

```r
naColumns <- colSums(is.na(trainingData)) > nrow(trainingData) * .5
trainingData <- trainingData[,!naColumns]
```
Now our training data has 55 columns.

Lastly, let's check for columns with near zero variance and remove those columns, too.

```r
nearZeroVarColumns <- nearZeroVar(trainingData)
trainingData <- trainingData[, -nearZeroVarColumns]
```
Now, our training data has 54 columns.

We should do the same steps on our testing data, too. 

```r
testingData <- testingData[,!naColumns]
testingData <- testingData[, -nearZeroVarColumns]
```
Now, our testing data also has 54 columns. We're satisfied with the cleaning part so far, and we're good to go :)

##Cross Validation
In order to avoid overfitting, it's good to do cross validation on our training data. Also, we would like to see our model's accuracy on a cross validation set, before we apply it on the final testing set. 

So we'll split our cleaned training data into two sets, named: cleanTraining and crossValidation.


```r
inTrain <- createDataPartition(trainingData$classe, p=0.7, list=FALSE)
cleanTraining <- trainingData[inTrain,]
crossValidation <- trainingData[-inTrain,]
```
Our clean training set has 13737 observations and our cross validation set has 5885 observations.

##Fitting a model
We'll build a random forest model because we're given a non-linear problem and we know that tree methods usually work well with high accuracy, with these kinds of problems. And, compared to a single tree model, random forest's accuracy would be better.


```r
model = randomForest(classe ~., data=cleanTraining)
model
```

```
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
```

Our OOB error rate of .34% looks promising and tells us that we can continue using this model.

Now it's time we get our predictions:

```r
prediction <- predict(model, crossValidation)
confMatrix <- confusionMatrix(prediction,crossValidation$classe)
outOfSampleError <- 1-sum(diag(confMatrix$table))/sum(confMatrix$table)
outOfSampleError
```

```
## [1] 0.002209006
```
Out of sample error is calculated as 0.0022.

Let's look at our confusion matrix in detail:

```r
print(confMatrix)
```

```
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
```
Our model's accuracy is 0.9978 which is pretty good :)

#Getting final predictions on the test set
Now we can apply our prediction model to the testing data with 20 observations.

```r
testSetPrediction <- predict(model, testingData)
testSetPrediction
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

#Saving final predictions on to the files
We'll define the function we're given by the Coursera team, to write the predictions in separate files

```r
pml_write_files = function(x){
n = length(x)
for(i in 1:n){
filename = paste0("problem_id_",i,".txt")
write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
}
}
```

Now, let's call this function with our testSetPrediction vector

```r
pml_write_files(testSetPrediction)
```

##Final validation
We've submitted these predictions to Coursera and our model passed the validation %100!
