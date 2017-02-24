###########################################################
### Logistic Regression Model
### Data Set: Framingham Heart Study (4240 Subset)
##########################################################

### Step 1: Split the data into training and testing data

library(dplyr)
library(ggplot2)
library(caTools)
library(tidyr)
library(ROCR)
library(Hmisc)
library(statisticalModeling)
library(ROCR)
library(rpart.plot)

### Step 2: Load data and view summaries

framingham = read.csv("framingham_edx.csv")
fhs = framingham
#str(fhs)
#summary(fhs)

### Step 3: Determine statistically significant Risk factors

summary(glm(TenYearCHD ~ ., data = fhs, family = binomial))

### Step 3: Create subset of data with only relevant variables

fhs_sub = fhs[c("TenYearCHD", "age", "male", "totChol", 
                "sysBP", "cigsPerDay", "glucose")]

### Step 3: Remove missing values 

fhs_sub = na.omit(fhs_sub)

### Step 5: Factorize categorical variables

fhs_sub$male = as.factor(fhs_sub$male)

### Generate Correlation Matrix for numeric values

fhs_sub_num = fhs_sub[c("TenYearCHD", "age", "totChol", "sysBP", "cigsPerDay", "glucose")]
p = cor(fhs_sub_num)
corrplot(p)

fhs = scale(fhs[])

### Step 4: Generate graphs for descriptive statistics

hist(fhs_sub$age, xlim = c(30,70), breaks = 25)
hist(fhs_sub$BMI, xlim= c(10,50), breaks=25)
hist(fhs_sub$glucose, xlim = c(40,400), breaks=50)

boxplot(glucose~male, data=fhs)
boxplot(BMI~male, data=fhs)
boxplot(totChol~male, data=fhs)
boxplot(sysBP~male, data=fhs)


ggplot(fhs_sub, aes(age, glucose)) + geom_point(alpha=0.2) + 
            geom_smooth(se=FALSE) + facet_grid(.~male)
ggplot(fhs_sub, aes(age, sysBP)) + geom_point(alpha=0.2) + 
              geom_smooth(se=FALSE) + facet_grid(.~male)
ggplot(fhs_sub, aes(age, totChol)) + geom_point(alpha=0.2) + 
              geom_smooth(se=FALSE) + facet_grid(.~male)
ggplot(fhs_sub, aes(age, sysBP)) + geom_point(alpha=0.2) + 
                 geom_smooth(se=FALSE) + facet_grid(.~male)

counts <- table(fhs_sub$TenYearCHD, fhs_sub$male)
barplot(counts, main="CHD by Gender", xlab="Gender", col=c("darkblue","red"))

### Step 5: Split data set into 'train' and 'test' subset

set.seed(1000)
split = sample.split(fhs_sub$TenYearCHD, SplitRatio = 0.65) 
fhs_train = subset(fhs_sub, split==TRUE) 
fhs_test = subset(fhs_sub, split!=TRUE)

### Step 5: Fit a logistic model to do predictions for test data

base_mod = glm(TenYearCHD ~ age + male + sysBP + totChol, 
               data = fhs_train, family = binomial)
summary(base_mod)

aug_mod = glm(TenYearCHD ~ age + male + sysBP + totChol + glucose, 
                 data = fhs_train, family = binomial)
summary(aug_mod)

base_mod_pred = predict(base_mod, type = "response", newdata = fhs_test)
table(fhs_test$TenYearCHD, base_mod_pred > 0.5)

aug_mod_pred = predict(aug_mod, type = "response", newdata = fhs_test)
table(fhs_test$TenYearCHD, aug_mod_pred > 0.5)

### Step 5 Create the confusion matrix, and compute the misclassification rate

base_mod_pred = evaluate_model(base_mod, data=fhs_test)
with(data= base_mod_pred, mean(TenYearCHD - model_output, na.rm = TRUE)^2)

aug_mod_pred = evaluate_model(aug_mod, data=fhs_test)
with(data= aug_mod_pred, mean(TenYearCHD - model_output, na.rm = TRUE)^2)

cv_pred_error(base_mod, aug_mod)

### Calculate AUC (Differientiate between low risk and high risk patients)

ROCRbase_mod = prediction(base_mod_pred, fhs_test$TenYearCHD)
as.numeric(performance(ROCRbase_mod, "auc")@ y.values)

ROCRaug_mod = prediction(aug_mod_pred, fhs_test$TenYearCHD)
as.numeric(performance(ROCRaug_mod, "auc")@ y.values)

### Plot AIC

plot(aicplot(TenYearCHD~age,male,sysBP,glucose,totChol, 
             data=fhs, family="binomial", alpha=seq(0.2,1.0,by=0.05)))

### Plot ROC Curves

ROCRperf_base_mod= performance(ROCRbase_mod, "tpr", "fpr")
plot(ROCRperf_base_mod, colorize = TRUE, print.cutoffs.at=seq(0.1,0.1), text.adj=c(-02, 1.7))

ROCRperf_aug_mod= performance(ROCRbase_mod, "tpr", "fpr")
plot(ROCRperf_aug_mod, colorize = TRUE, print.cutoffs.at=seq(0.1,0.1), text.adj=c(-02, 1.7))

### Recursive Partioning and Regression Tree

rpart_mod = rpart(TenYearCHD ~ ., cp = 0.005, data = fhs_sub)
prp(rpart_mod, type = 3)
