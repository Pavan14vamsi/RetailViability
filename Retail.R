xtest = read.csv("store_test.csv", stringsAsFactors = F)
xtrain = read.csv("store_train.csv", stringsAsFactors = F)
glimpse(xtrain)
#lapply(xtrain, function(x) sum(is.na(x)))
library(dplyr)
xtest$store =NA
xtrain$data = "train"
xtest$data = "test"
x = rbind(xtest,xtrain)
x$Id = NULL


lapply(x, function(x) sum(is.na(x)))
#Population anc country missing. removing from the training set and imputing in the test
lapply(x, function(x) sum(is.na(x)))
ind = which((is.na(x$country) & x$data=="test"), arr.ind = T) #The state code was ME
table(x$country[x$state_alpha=="ME"  &  x$countyname=="Cumberland County"]) #All the country codes which had the same state alpha and county name. Turns out it's five
x$country[ind] = 5

#Population missing in both, gonna remove the one in the train case and impute in test cae
lapply(x, function(x) sum(is.na(x)))
ind = which(is.na(x$population & x$data=="train"), arr.ind = T)
#View(x[ind,])
x = x[-ind,]
ind = which(is.na(x$population & x$data=="test"), arr.ind = T)
x$population[ind] = mean(x$population[-ind])
#All missing values taken care of


numericals = lapply(x, function(x) is.numeric(x))==T
numericals = names(numericals)[numericals]
numericals = numericals[-c(6,7,10)] #Remove country, state and store
(numericals)
x2 = x
#Remove outliers from trainingdata and Standardise them all
outliers = function(x){
  if(!is.numeric(x)){
    return()
  }
  q1 = quantile(x, 0.25)
  q3 = quantile(x,0.75)
  iqrange = IQR(x)
  upper = q3+1.5*iqrange
  lower = q1-1.5*iqrange
  ind = which((x>upper | x<lower), arr.ind = T)
  return(ind)
}



x = x2
outs = c()
v = x$data=="train"
xtrain = x%>% filter(data=="train")
xtest = x %>% filter(data=="test")
numericals
for(col in numericals){
  ind = outliers(xtrain[,col])
  outs = union(outs,ind)
}
xtrain = xtrain[-outs,] #All outliers from the training data removed

x = rbind(xtrain,xtest)
nrow(xtest)
mean(x$sales0)

for(col in numericals){
  mu = mean(x[,col])
  std = (var(x[,col]))**0.5
  x[,col] = (x[,col]-mu)/std
} #Standardised



x3 = x
categoricals = lapply(x, function(x) is.character(x))==T
categoricals = names(categoricals)[categoricals]
categoricals = union(categoricals, c("country", "state"))
categoricals = categoricals[-c(7,3,5)]
categoricals[6] = "State" #For some reaason S was was always lowercase


x$Areaname = NULL #Almost same as county town name
x$state_alpha = NULL #Same as state (in char form)

createDummies=function(data,var,freq_cutoff=100){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}
for(cat in categoricals){
  print(cat)  
  x = createDummies(x, cat)
}


#mean(x$population)
trainingdata = x %>% filter(data=="train") %>% select(-data)
testdata = x %>% filter(data=="test") %>% select(-data)
s = sample(1:nrow(trainingdata), 0.7*nrow(trainingdata))
validationdata = trainingdata[-s,]
trainingdata = trainingdata[s,]
#---------logistic regressor------------------#
library(car)

for_vif = lm(store ~ sales1 + sales2 + CouSub + population + country_19 + 
               country_27 + country_11 + country_3 + State_13 + State_33 + 
               State_50 + State_25 + State_23,data=trainingdata)

sort(vif(for_vif))
for_vif = step(for_vif)
formula(for_vif)
sort(summary(for_vif)$coefficients[,4])
log.regressor = glm(store ~ sales1 + sales2 + CouSub + population + country_19 + 
                      country_27 + country_11 + country_3 + State_13 + State_33 + 
                      State_50 + State_25 + State_23,
                    data = trainingdata, family="binomial")



val.actual = validationdata$store
validationdata$store = NULL
log.val.scores = predict(log.regressor, newdata = validationdata, type = "response")
range(log.val.scores)

cutoff=0.5
log.predicted = as.numeric(log.val.scores > cutoff)
predicted = log.predicted
TP = sum(val.actual==1 & predicted==1)
FP = sum(val.actual==0 & predicted==1)
TN = sum(val.actual==0 & predicted==0)
FN = sum(val.actual==1 & predicted==0)
P = TP + FN
N = TN + FP
Sn = TP/P
Sp = TN/N
dist = sqrt((1-Sp)**2 + (1-Sn)**2)
KS = Sn - (FP/N)
pROC::auc(log.predicted,val.actual)
accuracy = (TP + TN)/(P + N)
accuracy


#----------Random Forest---------------------#
library(randomForest)


param=list(mtry=c(5,10,15),
           ntree=c(50,100,200,500,700,1000, 2000),
           maxnodes=c(5,10,15,20,30,50,100,150),
           nodesize=c(1,2,5,10,15)
)
mycost_auc=function(y,yhat){
  roccurve=pROC::roc(y,yhat)
  score=pROC::auc(roccurve)
  return(1-score)
}
subset_paras=function(full_list_para,n=10){
  all_comb=expand.grid(full_list_para)
  s=sample(1:nrow(all_comb),n)
  subset_para=all_comb[s,]
  return(subset_para)
}


num_trials=30
my_params=subset_paras(param,num_trials)
#nrow(my_params)
library(cvTools)
myauc = 0
trainingdata$store = as.factor(trainingdata$store)
val.actual = as.factor(val.actual)
for(i in 1:num_trials){
  params = my_params[i,]
  k=cvTuning(randomForest,store~.,
             data =trainingdata,
             tuning =params,
             folds = cvFolds(nrow(trainingdata), K=10, type ="random"),
             cost =mycost_auc,
             predictArgs = list(type="prob")
  )
  score.this=k$cv[,2]
  if(score.this > myauc){
    best_params = params
    myauc = score.this
  }
}
1-myaucs
best_params

rf.classifier = randomForest(store~.,
                             data=trainingdata,
                             mtry=best_params$mtry, #The number of variables randomly collected each time
                             ntree=best_params$ntree+1300,
                             maxnodes=best_params$maxnodes,
                             nodesize=best_params$nodesize,
)

rf.val.scores = predict(rf.classifier, newdata = validationdata, type="prob")[,2]

cutoff=0.1
rf.predicted = as.numeric(rf.val.scores > cutoff)
predicted = rf.predicted
TP = sum(val.actual==1 & predicted==1)
FP = sum(val.actual==0 & predicted==1)
TN = sum(val.actual==0 & predicted==0)
FN = sum(val.actual==1 & predicted==0)
P = TP + FN
N = TN + FP
Sn = TP/P;Sn
Sp = TN/N;Sp
dist = sqrt((1-Sp)**2 + (1-Sn)**2)
KS = Sn - (FP/N)
val.actual = as.numeric(val.actual)
pROC::auc(predicted,val.actual)
accuracy = (TP + TN)/(P + N)
accuracy

#Maybe there is some class imbalance, maybe curing that will raise the auc score
class_0 = x %>% filter(store==0)
class_1 = x %>% filter(store==1)
nrow(xtrain)
nrow(class_1)

s = sample(1:nrow(class_1), 2000, replace = T)
s2 = sample(1:nrow(class_0), 2000, replace = T)
class_1.oversampled = class_1[s,]
class_0.oversamped = class_0[s2,]
setdiff(names(class_1.oversampled),names(trainingdata)  )
trainingdata.balanced = rbind(class_0.oversamped, class_1.oversampled)
trainingdata.balanced = trainingdata.balanced %>% select(-data)


s = sample(1:nrow(trainingdata.balanced), 0.7*nrow(trainingdata.balanced))
validationdata.balanced = trainingdata.balanced[-s,]
trainingdata.balanced = trainingdata.balanced[s,]

#GLM on balanced-----------------------------

for_vif2 = lm(store~.,data=trainingdata.balanced)
sort(vif(for_vif2))
for_vif2 = step(for_vif2)
formula(for_vif2)

lapply(trainingdata.balanced, function(x) is.factor(x))

log.regressor.balanced = glm(store ~ sales1 + sales2 + sales3 + sales4 + population + country_19 + 
                               country_15 + country_27 + country_9 + country_11 + country_3 + 
                               State_21 + State_13 + State_33 + State_50 + State_48 + State_25,
                               data = trainingdata.balanced, family="binomial")
log.val.scores.bal = predict(log.regressor.balanced, newdata = validationdata.balanced)

cutoff=0.2
log.val.bal.predicted = as.numeric(log.val.scores.bal > cutoff)
predicted = log.val.bal.predicted
val.actual.bal = validationdata.balanced$store
nrow(validationdata.balanced)
TP = sum(val.actual.bal==1 & predicted==1)
FP = sum(val.actual.bal==0 & predicted==1)
TN = sum(val.actual.bal==0 & predicted==0)
FN = sum(val.actual.bal==1 & predicted==0)
P = TP + FN
N = TN + FP
Sn = TP/P
Sp = TN/N
dist = sqrt((1-Sp)**2 + (1-Sn)**2)
KS = Sn - (FP/N)
val.actual = as.numeric(val.actual)
pROC::auc(predicted,val.actual.bal)
accuracy = (TP + TN)/(P + N)
accuracy
