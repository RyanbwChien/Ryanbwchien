
install.packages('devtools')
require(devtools)
install.packages('reticulate')
# install tensorflow(如果你要tensorflow的話)
devtools::install_github("rstudio/tensorflow") 
# installing keras(如果你要keras的話)
devtools::install_github("rstudio/keras") 
tensorflow::install_tensorflow(version = "https://github.com/mind/wheels/releases/download/tf1.5-gpu-cuda91-nomkl/tensorflow-1.5.0-cp27-cp27mu-linux_x86_64.whl")


packageVersion("tensorflow")
packageVersion("keras")

require(tensorflow)
sess = tf$Session()
hello <- tf$constant('Hello, TensorFlow!')
sess$run(hello)
Sys.getenv()

library(caret)
library(tensorflow)
library(keras)

library(readxl)
data<-read_excel("Cardiotocographic.xls")
table(data[22]) #table univariant count

#Change to matrix

data<-as.matrix(data)
dimnames(data)=NULL

#Normalize ((Min-max normalization))
data[,1:21]<- normalize(data[,1:21])
data[,22]<-as.numeric(data[,22])-1

#Data partition-train(70%) & test(20%)
set.seed(1234)
ind=sample(2,nrow(data),replace=T,prob=c(0.7,0.3))
training=data[ind==1,1:21]
test=data[ind==2,1:21]
trainingtarget=data[ind==1,22]
testtarget=data[ind==2,22]

#One Hot Encoding
trainLabels=to_categorical(trainingtarget)
testLabels=to_categorical(testtarget)

print(testLabels)
str(trainLabels)

library(dplyr)
#Create sequential model
model=keras_model_sequential()
model%>% layer_dense(unit=8,activation = "relu",input_shape = c(21))%>%
         layer_dense(unit=3,activation = "softmax")
summary(model)

#Compile
model%>% compile(loss="categorical_crossentropy",optimizer="adam",metrics="accuracy") #需注意matrics=>metrics
plot(model)
#Fit model
history=model%>%fit(training,trainLabels,epoch=200,batch_size=32,validation_split=0.2,verbose=0)
history<-model%>%fit(training,trainLabels,epoch=200,batch_size=32,validation_split=0.2)
plot(history)

devtools::install_github("andrie/deepviz")
install.packages("deepviz")
library(deepviz)
library(magrittr)
model %>% plot_model()
plot_model(model )

#Evaluate model with test data
model1=model%>%evaluate(test,testLabels)

#Prediction & confusion matrix-test data
prob<-model%>%predict_proba(test)
pred<-model%>%predict_classes(test)
table1=table(Predicted=pred,Actual=testtarget)
cbind(prob,pred,testtarget)

total=sum(table(Predicted=pred,Actual=testtarget))
correct=sum(diag(table(Predicted=pred,Actual=testtarget)))
incorrect=total-correct
acc=correct/total

#Fine-turn model
table1
model1


library(neuralnet)
require(readxl)
setwd("C:/Users/user/Desktop/R practice")
data<-read_excel("Cardiotocographic.xls") 
require(keras)

dummy=to_categorical(as.numeric(data[[22]])-1) # 使用keras中to_categorical功能需要把-1才會是原始類別數量
colnames(dummy)=paste("NSP_",rownames(table(data[22])),sep="")　#paste中要加sep=""才會中間無間隔
data=cbind(data,dummy)
colnames(data)

data[["NSP"]]=as.factor(data[["NSP"]])
data["NSP"]=as.factor(data[["NSP"]])
data$NSP=as.factor(data$NSP)
sapply(data,class)

#直接將欄位
nn<-neuralnet(NSP~LB+AC+FM,data=data,
              hidden = c(2,2),       # 一個隱藏層：2個node 1stlayer:2node; 2ndlayer:2node
              learningrate = 0.01, # learning rate
              threshold = 0.01,    # partial derivatives of the error function, a stopping criteria
              stepmax = 5e5 )       # 最大的ieration數 = 500000(5*10^5))
plot(nn)

#每次跑結果都會不一樣
nn2<-neuralnet(NSP_1+NSP_2+NSP_3~LB+AC+FM,data=data,
                hidden = c(2),       # 一個隱藏層：2個node 1stlayer:2node; 2ndlayer:2node
                learningrate = 0.01, # learning rate
                threshold = 0.01,    # partial derivatives of the error function, a stopping criteria
                stepmax = 5e5 )       # 最大的ieration數 = 500000(5*10^5))
plot(nn2)



plot(data$LB,data$AC)
abline(model)

model=aov(factor(NSP)~LB+AC+FM,data=data)
model.matrix(model)
model$coefficients
summary.lm(model)
AIC(model)






