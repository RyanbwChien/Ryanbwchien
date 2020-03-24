install.packages("insuranceData")
library(insuranceData)
data(dataCar)
head(dataCar)
s
install.packages("glm2")
library(glm2)
summary(dataCar$numclaims)
aggregate(dataCar,numclaims,length)
dataCar$newnumclaims=dataCar$numclaims+1
dataCar$newnumclaims2=((dataCar$newnumclaims)^-2-1)*(1/-2)
sapply(dataCar,class)

  result=glm2( numclaims~agecat+area,family = poisson(),data=dataCar)
  result=glm2( numclaims~factor(agecat)+area,family = poisson,data=dataCar)
  result2=glm2( numclaims~factor(agecat)+area,family = gaussian(log),data=dataCar)
  result4=glm2( numclaims~factor(agecat)+area,family = gamma()  ,data=dataCar)
  result3=glm2( numclaims~factor(agecat)+area,family = quasipoisson(),data=dataCar)
  summary(result)
  summary(result2)
  summary(result3)
  par(mfcol=c(2,2))
  plot(result)
  
  Returnresult2=lm( numclaims~factor(agecat)+area,data=dataCar)
  Returnresult3=aov( numclaims~factor(agecat)+area,data=dataCar)
  boxcox(Returnresult2)
  summary.lm(Returnresult2)
  summary.aov(Returnresult2)
  library(dplyr)
  library(reshape2)
  dataCar%>%group_by(factor(agecat),area)%>%count(numclaims)
  dcast(dataCar,agecat+as.character(area)~numclaims,count)#no applicable method for 'groups' applied to an object of class "factor"
  
  bartlett.test(resid(Returnresult2)~dataCar$agecat+dataCar$area)
  bartlett.test(resid(Returnresult2),Ops.factor(dataCar$agecat,dataCar$area))
  ?ncv.test
  install.packages("car")
  library(car)
  ncvTest(Returnresult2)
  durbinWatsonTest(Returnresult2,method="normal")
  AIC(Returnresult2)
  
  TukeyHSD(Returnresult3) #no applicable method for 'TukeyHSD' applied to an object of class "c('glm', 'lm')"
  library(multcomp)
  glht(Returnresult2,linfct=mcp(agecat="Tukey"))
  
  dataCar$agecat=as.factor(dataCar$agecat)
  hist(dataCar$numclaims)

  
  depars

  glmmodel=c("binomial",	"gaussian","Gamma","inverse.gaussian","poisson","quasi",	"quasibinomial"	,"quasipoisson")
  glmmodel=c(	"gaussian","Gamma","inverse.gaussian","poisson","quasi"	,"quasipoisson")
  glmmodel=c(	"gaussian","poisson","quasi",	"quasipoisson")

i=1
goodness=c()
expand.grid(levels(factor(dataCar$agecat)),levels(dataCar$area))

significant= data.frame(Date=as.Date(character()),
                             File=character(), 
                             User=character(), 
                             stringsAsFactors=FALSE) 
    
dataCar$newnumclaims
tryCatch({for(name in glmmodel ){
    model=glm2( numclaims~factor(agecat)+area,family = name,data=dataCar)
    result=paste("fit",name, sep=".")
    assign(paste("fit",name, sep="."),model)
    goodness[name]=AIC(model)
    i=i+1
    }},    
    warning = function(msg) {
      message("Original warning message:")
      message(paste0(msg,"\n"))
      return(NULL)},
    # 遇到 error 時的自訂處理函數
    error = function(msg) {
      message("Original error message:")
      message(paste0(msg,"\n"))
      return(NA)
    }
    )
fit.poisson
plot(model)

hist(rexp(100,0.001))

fit.quasibinomial
plot(fit.Gamma)
plot(fit.gaussian)
boxcox(fit.gaussian)
boxcox(fit.gaussian)$x[which.max(boxcox(fit.gaussian)$y)]

library(nortest)
ad.test(dataCar$newnumclaims2)
hist(dataCar$newnumclaims2)


hist(predict(fit.gaussian))
plot(fit.inverse.gaussian)
hist(predict(fit.inverse.gaussian))
plot(fit.quasipoisson)
fit.poisson=glm2( numclaims~factor(agecat)+area,family = "poisson",data=dataCar)
TukeyHSD(fit.poisson) #no applicable method for 'TukeyHSD' applied to an object of class "c('glm', 'lm')"
par(mfcol=c(2,2))
plot(fit.poisson)
hist(predict(fit.poisson))
hist()
predict(fit.poisson)
abline(h=3)

#"mixed effect model". Check out the lme4 package.and GLM model
library(lme4)
model5=glmer( Petal.Length~Sepal.Width+factor(Species)+1|Petal.Width,family = gaussian(),data=iris)


#GLS regression(Data correlation)
library(nlme)
gls.norm=gls(numclaims~factor(agecat)+area,correlation = corAR1(form= ~exposure),data=dataCar)


gls.norm1=gls(Petal.Length~Sepal.Width,data=iris) #Defalut 

gls.norm2=gls(Petal.Length~Sepal.Width,correlation = corARMA(p=1,q=4),data=iris)
lmmodel1=lm(Petal.Length~Sepal.Width,data=iris)
gls.norm3=gls(Petal.Length~Sepal.Width,correlation = corARMA(i=1,q=1),data=iris)

summary(gls.norm3)$coefficients
coef(summary(gls.norm))[,'p-value']
coef(gls.norm)

summary(gls.norm3)

install.packages('tseries') #必要套件
require(tseries)
lmres=summary(lmmodel1)$res

require(forecast) 
require(prophet) 
acf(lmres,lag.max=300)
par(mfrow=c(2,1))
plot(ts(lmres))
plot(ts(diff(lmres,1)))
pacf(ts(diff(lmres,1)),300)
acf(ts(diff(lmres,1)),300)

length(lmres)
adf.test(lmres)
auto.arima(ts(lmres),seasonal = T,test="adf",ic="bic") # package is forecast
auto.arima(ts(diff(lmres,1)),seasonal = T,test="adf",ic="aic")

arima3a <- arima(iris$Petal.Length,xreg = iris$Sepal.Width,order=c(2,0,1))


#多項式 regression
require(nnet)
test <- multinom(numclaims~factor(agecat)+area, data = dataCar)
summary(test)
hist(as.numeric(predict(test)))
plot(predict(test),dataCar$numclaims)
boxplot(as.numeric(predict(test))  )


#Step regression
step(fit.gaussian)
summary(step(fit.gaussian),k=log(nrow(dataCar)),method="both")



#計算T檢定值
coef(summary(test))/summary(test)$standard.errors

#計算T檢定值
p.vlaues=pnorm(abs(coef(summary(test))/summary(test)$standard.errors),lower.tail = F)*2


test$coef
segments(1,5,7,10)

confint(fit.poisson)

plot(fitted(fit.poisson)~dataCar[["numclaims"]])
plot(fitted(model)~dataCar[["numclaims"]])
summary(fit.gaussian)$coeff[,4]<0.05
summary(x.1)
x.2  
summary(x.2)
x.3
x.4
par(mfrow=c(2,2))
plot(fit.gaussian)

summary(fitdistr(dataCar$numclaims,"poisson"))

install.packages("fitdistrplus")
library(fitdistrplus)
fit.norm=fitdist(dataCar$numclaims,distr = "norm", method = "mme")
summary(fit.norm)
plot(fit.norm)


fitall=descdist(dataCar$numclaims)
summary(fitall)

shapiro.test(dataCar$numclaims)

install.packages("nortest")
library(nortest)
ad.test(dataCar$numclaims)

fit.norm2




