#Attaching packages

library(tidyverse)
library(mlbench)
library(gmodels)
library(class)

train.path <- "/kaggle/input/banking-dataset-marketing-targets/train.csv"
test.path <- "/kaggle/input/banking-dataset-marketing-targets/test.csv"

Mt_df <- read.table(train.path,header = T, sep = ";")
test_df <- read.table(test.path,header = T, sep = ";")
head(Mt_df)

str(Mt_df)

fact <- c(2:5,7:9,11,16,17)
num <- c(1,6,10,12:15)

for (i in fact)
{
  print(names(Mt_df)[i])
  print("Train Data(%):")
  print(round(prop.table(table(Mt_df[,i]))*100,1))
  print("Test Data(%):")
  print(round(prop.table(table(test_df[,i]))*100,1))
}

for (i in num)
{
  print(names(Mt_df)[i])
  print("Train Data(%):")   
  print(summary(Mt_df[,i]))
  print("Test Data(%):")
  print(summary(test_df[,i]))
}

#Checking for NA and Null values

any(is.null(Mt_df))
any(is.na(Mt_df))

#Visualizations of the data and analysis

#Barplot
y_bar <- ggplot(Mt_df, aes(y))
y_bar + geom_bar(color = "black",fill = "blue") + theme(text = element_text(size=30))

#Histogram
fig(12, 8)
age_hist <- ggplot(Mt_df, aes(age))
age_hist + geom_histogram(binwidth = 5, color = "black",fill = "green") + theme(text = element_text(size=30))


summary(Mt_df$age)

fig(16, 8)
job_bar <- ggplot(Mt_df, aes(job))
job_bar + geom_bar(color = "black",fill = "purple") +  theme(text = element_text(size=30), axis.text.x=element_text(angle = 90, vjust = 0.5, hjust=1,size=30))

round(prop.table(table(Mt_df$job))*100,1)

fig(10, 8)
marital_bar <- ggplot(Mt_df, aes(marital))
marital_bar + geom_bar(color = "black",fill = "darkturquoise") + theme(text = element_text(size=30))

round(prop.table(table(Mt_df$marital))*100,1)

fig(10, 8)
education_bar <- ggplot(Mt_df, aes(education))
education_bar + geom_bar(color = "black",fill = "coral") + theme(text = element_text(size=30))

round(prop.table(table(Mt_df$education))*100,1)

default_bar <- ggplot(Mt_df, aes(default))
default_bar + geom_bar(color = "black",fill = "khaki") + theme(text = element_text(size=30))

round(prop.table(table(Mt_df$default))*100,1)

fig(20, 8)
balance_area <- ggplot(Mt_df, aes(balance))
balance_area + geom_area(stat = "bin", color = "black",fill = "cyan2",alpha = 0.5) + theme(text = element_text(size=30))

summary(Mt_df$balance)

fig(8,10)
housing_bar <- ggplot(Mt_df, aes(housing))
housing_bar + geom_bar(color = "black",fill = "firebrick2") + theme(text = element_text(size=30))

round(prop.table(table(Mt_df$housing))*100,1)

fig(8,10)
loan_bar <- ggplot(Mt_df, aes(loan))
loan_bar + geom_bar(color = "black",fill = "darkviolet") + theme(text = element_text(size=30))

round(prop.table(table(Mt_df$loan))*100,1)

fig(10,10)
contact_bar <- ggplot(Mt_df, aes(contact))
contact_bar + geom_bar(color = "black",fill = "forestgreen") + theme(text = element_text(size=30))

round(prop.table(table(Mt_df$contact))*100,1)

fig(20, 8)
day_hist <- ggplot(Mt_df, aes(day))
day_hist + geom_area(stat = "bin", color = "black",fill = "slateblue1", alpha = 0.5) + theme(text = element_text(size=30))

summary(Mt_df$day)

fig(16, 8)
Mt_df$month <- factor(Mt_df$month,levels = c("jan", "feb", "mar", "apr","may", "jun", "jul", "aug","sep", "oct", "nov", "dec")) #sort by chronological month order
month_bar <- ggplot(Mt_df, aes(month))
month_bar + geom_bar(color = "black",fill = "gold") +  theme(text = element_text(size=30), axis.text.x=element_text(angle = 90, vjust = 0.5, hjust=1,size=30))

round(prop.table(table(Mt_df$month))*100,1)

fig(20, 8)
duration_area <- ggplot(Mt_df, aes(duration))
duration_area + geom_area(stat = "bin", color = "black",fill = "pink2", alpha = 0.5) + theme(text = element_text(size=30))

summary(Mt_df$duration)

fig(20, 8)
campaign_hist <- ggplot(Mt_df, aes(campaign))
campaign_hist + geom_histogram(binwidth = 2, color = "black",fill = "chocolate2") + theme(text = element_text(size=30))

summary(Mt_df$campaign)

fig(20, 8)
pdays_area <- ggplot(Mt_df, aes(pdays))
pdays_area + geom_area(binwidth = 10, stat = "bin" ,alpha = 0.5, color = "black",fill = "deepskyblue3") + theme(text = element_text(size=30))

summary(Mt_df$pdays)









#Statistical Models For Classification

#K Nearest Neighbors

#preprocessing
knn.df <- Mt_df
head(knn.df)

for (i in 1:16)
{
  knn.df[,i] <- as.numeric(knn.df[,i])
}
knn.df$pdays[knn.df$pdays==-1]=0 
knn.df$previous[knn.df$previous==-1]=0 
head(knn.df)

knn.df1 <- sapply(knn.df[,1:16],scale)
knn.df1 <- as.data.frame(knn.df1)
knn.df <- cbind(knn.df1,knn.df$y)
head(knn.df)

#feature selection

# significance level using linear regression
knn.df1 <- knn.df 
knn.df1[,17] <- as.numeric(knn.df1[,17])
for (i in 1:16)
{
  print(names(knn.df1)[i]) 
  print(summary(lm(knn.df1[,17]~knn.df1[,i],knn.df1)))
}

set.seed(8)
train.size = floor(0.75*nrow(knn.df))
train.index = sample(1:nrow(knn.df), train.size)
train.set = knn.df[train.index,]
test.set = knn.df[-train.index,]

x.train = train.set[,-17] 
x.test = test.set[,-17] 
y.train = train.set[,17] 
y.test = test.set[,17] 

knn.3 <- knn(train = x.train, test = x.test, cl = y.train , k = 3)
TB = table(predicted = knn.3, true = y.test)
accuracy = round((TB[1]+TB[4])/sum(TB)*100,2)
accuracy





library(caret)

#confusionMatrix

lvs <- c("no", "yes")
truth <- factor(rep(lvs, times = c(9940, 1363)),
                levels = rev(lvs))
pred <- factor(
  c(
    rep(lvs, times = c(9539, 401)),
    rep(lvs, times = c(853, 510))),
  levels = rev(lvs))

caret::confusionMatrix(pred, truth)


#Logistic Regression
lr.df <- Mt_df
head(lr.df)

library(fastDummies)
lr.df = dummy_cols(lr.df, select_columns = c("job","marital","education","contact","month","poutcome"))
head(lr.df)

# significance level using linear regression
lr.df1 <- lr.df 
lr.df1[,17] <- as.numeric(lr.df1[,17])
for (i in 18:55)
{
  print(names(lr.df1)[i]) 
  print(summary(lm(lr.df1[,17]~lr.df1[,i],lr.df1)))
}

#Decision Tree

dt.df <- Mt_df
head(dt.df)

library(tree)
set.seed(8)
train.size = floor(0.75*nrow(dt.df))
train.index = sample(1:nrow(dt.df), train.size)
train.set = dt.df[train.index,]
test.set = dt.df[-train.index,]
y.test = dt.df[-train.index,17]
dt.model=tree(y~.,dt.df, subset = train.index)
summary(dt.model)

plot(dt.model)
text(dt.model,pretty=0,cex = 2)

Prediction.dt=predict(dt.model,test.set,type="class")
table(predicted = Prediction.dt, true = y.test)


#Confusion Matrix and Statistics

lvs <- c("no", "yes")
truth <- factor(rep(lvs, times = c(9940, 1363)),
                levels = rev(lvs))
pred <- factor(
  c(
    rep(lvs, times = c(9409, 531)),
    rep(lvs, times = c(725, 638))),
  levels = rev(lvs))

caret::confusionMatrix(pred, truth)


Prediction.dt2=predict(prune_dt.model,test.set,type="class")
table(predicted = Prediction.dt2, true = y.test)

#Random Forest

library(randomForest)
set.seed(8)
rf.df <- Mt_df
train.size = floor(0.75*nrow(rf.df))
train.index = sample(1:nrow(rf.df), train.size)
train.set = rf.df[train.index,]
test.set = rf.df[-train.index,]
y.test = rf.df[-train.index,17]
rf.model <- randomForest(y ~ .,data=train.set)
Prediction.rf = predict(rf.model, newdata=test.set)
table(predicted = Prediction.rf, true = y.test)


print(rf.model)
#Evaluate variable importance
importance(rf.model)
varImpPlot(rf.model,cex = 2)


caret::confusionMatrix(Prediction.rf, y.test,positive="yes")

table <- data.frame(caret::confusionMatrix(Prediction.rf, y.test,positive="yes")$table)

plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "good", "bad")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))


ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1,size = 30) +
  scale_fill_manual(values = c(good = "forestgreen", bad = "firebrick1")) +
  theme_bw() +
  xlim(rev(levels(table$Reference))) +
  theme(text = element_text(size=30)) +
  xlab("True Value")


library(randomForest)
set.seed(8)
rf.finalmodel <- randomForest(y ~ .,data=rf.df)
Predictionfinal = predict(rf.finalmodel, newdata=test_df)
table(predicted = Predictionfinal, true = test_df[,17])

caret::confusionMatrix(Predictionfinal, test_df[,17],positive="yes")

table <- data.frame(caret::confusionMatrix(Predictionfinal, test_df[,17],positive="yes")$table)

plotTable <- table %>%
  mutate(goodbad = ifelse(table$Prediction == table$Reference, "good", "bad")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1,size = 30) +
  scale_fill_manual(values = c(good = "forestgreen", bad = "firebrick1")) +
  theme_bw() +
  xlim(rev(levels(table$Reference))) +
  theme(text = element_text(size=30)) +
  xlab("True Value")