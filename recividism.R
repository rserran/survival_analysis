# An illustration: Recividism
# Source: Appendix-Cox-Regression,pdf (3.2)

# load packages
library(carData)
library(survival)

# laod Rossi dataset
data("Rossi")
head(Rossi[, 1:10])

unique(Rossi$week)

# Cox PH regression model
mod.allison <- coxph(Surv(week, arrest) ~ fin + age + race + wexp + 
                          mar + paro + prio + educ, 
                     data = Rossi)

mod.allison

summary(mod.allison)

# ANOVA
library(car)
Anova(mod.allison)

# graphical test of PH
library(survminer)

ggcoxzph(cox.zph(mod.allison))

# forest plot
ggforest(mod.allison)

# Cox PH diagnostics
ggcoxdiagnostics(mod.allison, type = 'deviance')

ggcoxdiagnostics(mod.allison, type = 'martingale')

# save Rossi subset to csv file
write.csv(Rossi[, 1:10], './rossi_select_df.csv')

# heart failure dataset
data("heart")
head(jasa)
