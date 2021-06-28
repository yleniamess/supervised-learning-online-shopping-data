library(devtools)       # to install beta version update of neuralnet
devtools::install_github("bips-hb/neuralnet") # Neuralnet library 1.44.6 
library(DataExplorer)   # exploratory data analysis
library(ggplot2)        # create data visualizations
library(gridExtra)      # arrange multiple grid-based plots on a page
library(ROCR)           # ROC curve
library(e1071)          # support vector machines
library(neuralnet)      # neural networks
library(nnet)

train_data <- read.csv("online_shoppers_intention_train.csv")
train_data <- train_data[,-1] # delete first column "X" which is useless

validation_data <- read.csv("online_shoppers_intention_validation.csv")
validation_data <- validation_data[,-1] 

# DESCRIPTIVE ANALYSIS ----

str(train_data) 

sum(complete.cases(train_data)) # check for missing values
sum(!complete.cases(train_data))

introduce(train_data)           # describe basic information for input data
plot_intro(train_data)          # plot basic information (from introduce) for input data

## Univariate Analysis
create_report(train_data)

## Bivariate Analysis
attach(train_data)

# Numerical - Categorical
p1 <- ggplot(train_data, aes(x = Administrative, fill = Revenue)) + 
  geom_density(alpha = 0.3)

p2 <- ggplot(train_data, aes(x = Administrative_Duration, fill = Revenue)) +
  geom_density(alpha = 0.3)

p3 <- ggplot(train_data, aes(x = Informational, fill = Revenue)) +
  geom_density(alpha = 0.3)

p4 <- ggplot(train_data, aes(x = Informational_Duration, fill = Revenue)) +
  geom_density(alpha = 0.3)

p5 <- ggplot(train_data, aes(x = ProductRelated, fill = Revenue)) +
  geom_density(alpha = 0.3) 

p6 <- ggplot(train_data, aes(x = ProductRelated_Duration, fill = Revenue)) +
  geom_density(alpha = 0.3)

p7 <- ggplot(train_data, aes(x = BounceRates, fill = Revenue)) +
  geom_density(alpha = 0.3)

p8 <- ggplot(train_data, aes(x = ExitRates, fill = Revenue)) +
  geom_density(alpha = 0.3)

p9 <- ggplot(train_data, aes(x = PageValues, fill = Revenue)) +
  geom_density(alpha = 0.3)

p10 <- ggplot(train_data, aes(x = SpecialDay, fill = Revenue)) +
  geom_density(alpha = 0.3)

grid.arrange(p1,p2,p3,p4,p5,p6, nrow = 3, ncol = 2, top = "Conditional distributions given Revenue")
grid.arrange(p7,p8,p9,p10, nrow = 2, ncol = 2)

# Categorical - Categorical
p11 <- ggplot(train_data, aes(x = Month, fill = Revenue)) + 
  geom_bar(position = "fill") +
  scale_fill_brewer(palette = "Pastel2") +
  ylab("proportion") +
  theme_minimal()

p12 <- ggplot(train_data, aes(x = factor(OperatingSystems), fill = Revenue)) + 
  geom_bar(position = "fill") +
  scale_fill_brewer(palette = "Pastel2") +
  xlab("Operating Systems") +
  ylab("proportion") +
  theme_minimal()

p13 <- ggplot(train_data, aes(x = factor(Browser), fill = Revenue)) + 
  geom_bar(position = "fill") +
  scale_fill_brewer(palette = "Pastel2") +
  xlab("Browser") +
  ylab("proportion") +
  theme_minimal()

p14 <- ggplot(train_data, aes(x = factor(Region), fill = Revenue)) + 
  geom_bar(position = "fill") +
  scale_fill_brewer(palette = "Pastel2") +
  xlab("Region") +
  ylab("proportion") +
  theme_minimal()

p15 <- ggplot(train_data, aes(x = factor(TrafficType), fill = Revenue)) + 
  geom_bar(position = "fill") +
  scale_fill_brewer(palette = "Pastel2") +
  xlab("Traffic Type") +
  ylab("proportion") +
  theme_minimal()

p16 <- ggplot(train_data, aes(x = VisitorType, fill = Revenue)) + 
  geom_bar(position = "fill") +
  scale_fill_brewer(palette = "Pastel2") +
  xlab("Visitor Type") +
  ylab("proportion") +
  theme_minimal()

p17 <- ggplot(train_data, aes(x = Weekend, fill = Revenue)) + 
  geom_bar(position = "fill") +
  scale_fill_brewer(palette = "Pastel2") +
  ylab("proportion") +
  theme_minimal()

grid.arrange(p11,p12,p13,p14,p15,p16,p17, nrow = 4, ncol = 2, top = "Conditional distributions given Revenue")

# LOGISTIC REGRESSION ----

## Data Preparation:
# Converting to Factors
train_data$OperatingSystems <- factor(train_data$OperatingSystems)
train_data$Browser <- factor(train_data$Browser)
train_data$Region <- factor(train_data$Region)
train_data$TrafficType <- factor(train_data$TrafficType)

## Performing Variable Selection:
# Stepwise Logistic Regression:
logR <- step(glm(Revenue~., data = train_data, family = binomial), direction = "both")

## Prediction
summary(logR)

glm.probs = predict(logR,type = "response")
glm.pred = rep("FALSE", dim(train_data)[1])
glm.pred[glm.probs > 0.5] = "TRUE"
conf_matrix_train = addmargins(table(glm.pred, train_data$Revenue, dnn = c("Predicted Class","True Class")))
conf_matrix_train

misc_logR_train = round((conf_matrix_train[1,2] + conf_matrix_train[2,1])/conf_matrix_train[3,3] * 100, 2)
misc_logR_train # Misclassification error on training set (%)

tpr = round(conf_matrix_train[2,2]/conf_matrix_train[3,2]*100, 2) # TPR (%)
tpr

fpr = round(conf_matrix_train[2,1]/conf_matrix_train[3,1]*100, 2) # FPR (%)
fpr

round((100 - tpr), 2) # false negative error (%)

# ROC Curve
logR_performance = performance(prediction(glm.probs, train_data$Revenue), measure = "tpr", x.measure = "fpr")
plot(logR_performance, colorize = TRUE, lwd = 2, print.cutoffs.at = c(0.2,0.3,0.4,0.5,0.8))
abline(a = 0, b = 1, lty = 2)
logR_auc = performance(prediction(glm.probs, train_data$Revenue), measure = "auc", x.measure = "fpr")@y.values[[1]]
logR_auc

# Validation data
sum(complete.cases(validation_data))
sum(!complete.cases(validation_data))

validation_data$OperatingSystems <- factor(validation_data$OperatingSystems)
validation_data$Browser <- factor(validation_data$Browser)
validation_data$Region <- factor(validation_data$Region)
validation_data$TrafficType <- factor(validation_data$TrafficType)

glm.probs.val = predict(logR, validation_data, type = "response")
glm.pred.val = rep("FALSE", dim(validation_data)[1])
glm.pred.val[glm.probs.val > 0.5] = "TRUE"
conf_matrix_val = addmargins(table(glm.pred.val, validation_data$Revenue, dnn = c("Predicted Class","True Class")))
conf_matrix_val

misc_logR_val = round((conf_matrix_val[1,2] + conf_matrix_val[2,1])/conf_matrix_val[3,3] * 100, 2)
misc_logR_val

# SUPPORT VECTOR MACHINES ----
normalize <- function(x) {                    # function for MinMaxScaler
  return ((x - min(x)) / (max(x) - min(x)))
}

train_data2 = train_data  # make a copy of data
train_data2$ProductRelated_Duration = normalize(train_data2$ProductRelated_Duration)
train_data2$ExitRates = normalize(train_data2$ExitRates)
train_data2$PageValues = normalize(train_data2$PageValues)

validation_data2 = validation_data
validation_data2$ProductRelated_Duration = normalize(validation_data2$ProductRelated_Duration)
validation_data2$ExitRates = normalize(validation_data2$ExitRates)
validation_data2$PageValues = normalize(validation_data2$PageValues)

# Support Vector Classifier
set.seed(1)

tune.out = tune(svm, Revenue~ProductRelated_Duration+ExitRates+PageValues+Month+VisitorType, data = train_data2, kernel = "linear",
                ranges = list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100)), type = "C-classification")

summary(tune.out)
bestmod = tune.out$best.model
summary(bestmod)

svc.pred = predict(bestmod, validation_data2)
svc.confM = addmargins(table(svc.pred, validation_data2$Revenue, dnn = c("Predicted Class","True Class")))
svc.confM

svc.misc.err = round((svc.confM[1,2] + svc.confM[2,1])/svc.confM[3,3] * 100, 2)
svc.misc.err

# Support Vector Machine: Radial & Polynomial
# Radial
set.seed(1)

tune.out2 = tune(svm, Revenue~ProductRelated_Duration+ExitRates+PageValues+Month+VisitorType, data = train_data2, kernel = "radial",
                ranges = list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100), gamma = c(0.5, 1, 2, 3, 4)),
                type = "C-classification")

summary(tune.out2)
bestmod2 = tune.out2$best.model
summary(bestmod2)

svmr.pred = predict(bestmod2, validation_data2)
svmr.confM = addmargins(table(svmr.pred, validation_data2$Revenue, dnn = c("Predicted Class","True Class")))
svmr.confM

svmr.misc.err = round((svmr.confM[1,2] + svmr.confM[2,1])/svmr.confM[3,3] * 100, 2)
svmr.misc.err

# Polynomial
set.seed(1)

tune.out3 = tune(svm, Revenue~ProductRelated_Duration+ExitRates+PageValues+Month+VisitorType, data = train_data2, kernel = "radial",
                 ranges = list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100), degree = c(2,3,4,5,6,7,8,9,10)),
                 type = "C-classification")

summary(tune.out3)
bestmod3 = tune.out3$best.model
summary(bestmod3)

svmp.pred = predict(bestmod3, validation_data2)
svmp.confM = addmargins(table(svmp.pred, validation_data2$Revenue, dnn = c("Predicted Class","True Class")))
svmp.confM

svmp.misc.err = round((svmp.confM[1,2] + svmp.confM[2,1])/svmp.confM[3,3] * 100, 2)
svmp.misc.err

svmtab = data.frame(Model = c("Support Vector Classifier", "Radial SVM", "Polynomial SVM"), Misc.Error = c(svc.misc.err, svmr.misc.err, svmp.misc.err))
names(svmtab) = c("SVM models", "Miscl. error on validation set (%)")
svmtab

bestsvm = svmtab[which.min(svmtab$`Miscl. error on validation set (%)`),]
bestsvm

# NEURAL NETWORKS ----

# [TRAINING DATA]:
train_data3 <- train_data2

# Code the dependent variable Revenue as 0-1
y = as.matrix(train_data3[,18])
y[which(y == "FALSE")] = 0
y[which(y == "TRUE")] = 1
y = as.numeric(y)
train_data3$Revenue = y

# Transforming categorical variables of interest
Month = class.ind(train_data3$Month)
VisitorType = class.ind(train_data3$VisitorType)

# Removing: Month & VisitorType - Rebuilding the data frame
m = which(colnames(train_data3)=="Month")
v = which(colnames(train_data3)=="VisitorType")

train_data3 = train_data3[, -(c(m,v))]
train_data3 = cbind(train_data3, Month, VisitorType)
str(train_data3)

# [VALIDATION SET]:
validation_data3 <- validation_data2

# Code the dependent variable Revenue as 0-1
y = as.matrix(validation_data3[,18])
y[which(y == "FALSE")] = 0
y[which(y == "TRUE")] = 1
y = as.numeric(y)
validation_data3$Revenue = y

# Transforming categorical variables of interest
Month = class.ind(validation_data3$Month)
VisitorType = class.ind(validation_data3$VisitorType)

# Removing: Month & VisitorType - Rebuilding the data frame
m = which(colnames(train_data3)=="Month")
v = which(colnames(train_data3)=="VisitorType")

validation_data3 = validation_data3[, -(c(m,v))]
validation_data3 = cbind(validation_data3, Month, VisitorType)
str(validation_data3)

# Train a neural network for classification: linear.output = F

NN = c()
NN_names = c()
nn.error_train = c()
nn.error_val = c()

set.seed(1)
for (i in 1:8){
  cat(paste("Computing NN model with ", i, "hidden neurons ..."), "\n" )
  nn_i = neuralnet(Revenue~ProductRelated_Duration+ExitRates+PageValues+
                     Aug+Dec+Feb+Jul+June+Mar+May+Nov+Oct+Sep+
                     New_Visitor+Other+Returning_Visitor,
                   data = train_data3, hidden = i,
                   linear.output = F, rep = 3, stepmax = 1e+07)
  
  NN = c(NN, list(nn_i))
  NN_names = c(NN_names, paste("nn", i, sep = ""))
  
  i_best = which.min(nn_i$result.matrix[1,])
  while (is.null(nn_i$net.result[[i_best]])==T){
    i_best = i_best + 1
  }
  yhat = round(nn_i$net.result[[i_best]])
  
  nn.confM_train = addmargins(table(yhat, train_data3$Revenue))
  err_train_i = round((nn.confM_train[1,2] + nn.confM_train[2,1])/nn.confM_train[3,3] * 100, 2)
  nn.error_train = c(nn.error_train, err_train_i)
  
  nn.pred = round(predict(nn_i, newdata = validation_data3, rep = i_best))
  nn.confM_val = addmargins(table(nn.pred, validation_data3$Revenue))
  err_val_i = round((nn.confM_val[1,2] + nn.confM_val[2,1])/nn.confM_val[3,3] * 100, 2)
  nn.error_val = c(nn.error_val, err_val_i)
}

names(NN) = NN_names

nn.overview = data.frame(model = names(NN), hidden = c(1:8), train.misc.error = nn.error_train, val.misc.error = nn.error_val)
names(nn.overview) = c("NN models", "Hidden Neurons", "Miscl. error on training set (%)", "Miscl. error on validation set (%)")
nn.overview

bestnn = nn.overview[which.min(nn.overview$`Miscl. error on validation set (%)`),]
bestnn

nn = NN[[bestnn[1,1]]]
i_best = which.min(nn$result.matrix[1,])
plot(nn, rep = i_best)

# BEST MODEL & PREDICTION ON TEST SET ----

models = data.frame(model = c("Logistic Regression", "Radial SVM", "Neural Network"), misc.error = c(misc_logR_val, bestsvm[1,2], bestnn[1,3]))
names(models) = c("Stat. Learning Models", "Miscl. error on validation set (%)")
models
bestmodel = models[which.min(models$`Miscl. error on validation set (%)`),]
bestmodel

test_data <- read.csv("online_shoppers_intention_test.csv")
test_data <- test_data[,-1]

str(test_data)
sum(complete.cases(test_data)) 
sum(!complete.cases(test_data))
plot_intro(test_data)

Month = class.ind(test_data$Month)
VisitorType = class.ind(test_data$VisitorType)
m = which(colnames(test_data)=="Month")
v = which(colnames(test_data)=="VisitorType")
test_data = test_data[, -(c(m,v))]
test_data = cbind(test_data, Month, VisitorType)

bestmodel = NN[[bestnn[1,1]]]
probs = predict(bestmodel, test_data)
predictedValues = rep("FALSE", dim(test_data)[1])
predictedValues[probs > 0.5] = "TRUE"

table(predictedValues)
round(table(predictedValues)/length(predictedValues)*100, 2)

predicted_values_id_number = data.frame(id_number = test_data[, which(colnames(test_data)=="id_number")], Revenue = predictedValues)
write.csv(predicted_values_id_number, "predicted_values_id_number.csv", row.names = F)
head(predicted_values_id_number)

# DATA ANALYSIS ACCORDING TO VARIABLE "MONTH" ----

test_data <- read.csv("online_shoppers_intention_test.csv")
test_data <- test_data[,-1]

test_data_with_prediction = data.frame(test_data, Revenue = predictedValues)

counts_table = table(test_data_with_prediction$Revenue, test_data_with_prediction$Month)
round(prop.table(counts_table, margin = 1)*100, 0)

ggplot(test_data_with_prediction, aes(x = Month, fill = Revenue)) + 
  geom_bar(position = "fill") +
  scale_fill_brewer(palette = "Pastel2") +
  ylab("proportion") +
  theme_minimal()

