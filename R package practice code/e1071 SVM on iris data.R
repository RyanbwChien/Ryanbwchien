library("e1071")
head(iris)

attach(iris)

# Divide Iris data to x (containt the all features) and y only the classes

x <- subset(iris, select=-Species)
y <- Species

#Create SVM Model and show summary
svm_model <- svm(Species ~ ., data=iris)
summary(svm_model)

svm_model1 <- svm(x,y)
summary(svm_model1)


pred <- predict(svm_model1,x)
system.time(pred <- predict(svm_model1,x))


#Tune SVM Model and show summary
svm_tune <- tune(svm, train.x=x, train.y=y, 
                 kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))

print(svm_tune)

svm_model_after_tune <- svm(Species ~ ., data=iris, kernel="radial", cost=1, gamma=0.5)
summary(svm_model_after_tune)

pred <- predict(svm_model_after_tune,x)
system.time(predict(svm_model_after_tune,x))

table(pred,y)




