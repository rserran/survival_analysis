# Random Survival Forest Analysis
# Source: https://rpubs.com/camposacrs/997049

# load packages
suppressMessages(library(tidyverse))
library(survival)
library(randomForestSRC)
theme_set(theme_minimal())

# load data
data("peakVO2")

peakVO2 %>% 
     glimpse()

skimr::skim_without_charts(peakVO2)

## Random Forest model
B <- 600

# Building a RSF
rf_obj <- rfsrc(Surv(ttodead, died) ~ ., peakVO2, ntree = B,  membership = TRUE, importance=TRUE)

# Printing the RF object  
print(rf_obj)

# plot error rate and variable importance
plot(rf_obj)

# Comment: Probably with 600 trees, we get the same OOB error rate

## Predicting new observations

# Creating an hypothetical observation 
newdata <- data.frame(lapply(1:ncol(rf_obj$xvar),function(i){median(rf_obj$xvar[,i])}))
colnames(newdata) <- rf_obj$xvar.names
newdata [,which(rf_obj$xvar.names == "peak_vo2")] <- quantile(rf_obj$xvar$peak_vo2, 0.25)

newdata

# generate prediction
y.pred <- predict(rf_obj,newdata = rbind(newdata, rf_obj$xvar)[1,])

y.pred$predicted

par(cex.axis = 1.0, cex.lab = 1.0, cex.main = 1.0, mar = c(6.0,6,1,1), mgp = c(4, 1, 0))
plot(round(y.pred$time.interest,2),y.pred$survival[1,], type="l", xlab="Time (Year)",   
     ylab="Survival", col=1, lty=1, lwd=2)

## Evaluate model

## obtain Brier score using KM censoring distribution estimators
bs.km <- get.brier.survival(rf_obj, cens.mode = "km")$brier.score

## plot the brier score
plot(bs.km, type = "s", col = 2, ylab="Brier Score")


## `ggrandomforest` package
library(ggRandomForests)

plot(gg_survival(interval = 'ttodead', censor = 'died', 
            data = peakVO2)
)

# stratified by `male` (gender)
plot(gg_survival(interval = 'ttodead', censor = 'died', 
                 data = peakVO2, by = 'male')
)

