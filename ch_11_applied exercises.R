# ISLR 2nd ed Chapter 11 - Survival Analysis and Censored Data
# Applied exercises
# Source: https://danhalligan.github.io/ISLRv2-solutions/survival-analysis-and-censored-data.html#question-10-7

# load packages
suppressMessages(library(tidyverse))
library(ISLR2)
theme_set(theme_bw())

# load dataset
data("BrainCancer")

BrainCancer

# data description https://rdrr.io/cran/ISLR2/man/BrainCancer.html

# first look
skimr::skim(BrainCancer)

# Q.10

# a. Plot the Kaplan-Meier survival curve with ±1 standard error bands, using the 
# survfit() unction in the survival package.
library(survival)

x <- Surv(BrainCancer$time, BrainCancer$status)
plot(survfit(x ~ 1),
     xlab = "Months",
     ylab = "Estimated Probability of Survival",
     col = "steelblue",
     conf.int = 0.67
)

# b. Draw a bootstrap sample of size n=88 from the pairs (yi, δi), and compute 
# the resulting Kaplan-Meier survival curve. Repeat this process B=200 times.
# Use the results to obtain an estimate of the standard error of the
# Kaplan-Meier survival curve at each timepoint. Compare this to the standard
# errors obtained in (a).

plot(survfit(x ~ 1),
     xlab = "Months",
     ylab = "Estimated Probability of Survival",
     col = "steelblue",
     conf.int = 0.67
)

fit <- survfit(x ~ 1)

dat <- tibble(time = c(0, fit$time))

for (i in 1:200) {
     y <- survfit(sample(x, 88, replace = TRUE) ~ 1)
     y <- tibble(time = c(0, y$time), "s{i}" := c(1, y$surv))
     dat <- left_join(dat, y, by = "time")
}

res <- fill(dat, starts_with("s")) |>
     rowwise() |>
     transmute(sd = sd(c_across(starts_with("s"))))

se <- res$sd[2:nrow(res)]
lines(fit$time, fit$surv - se, lty = 2, col = "red")
lines(fit$time, fit$surv + se, lty = 2, col = "red")

# c. Fit a Cox proportional hazards model that uses all of the predictors to
# predict survival. Summarize the main findings.

fit <- coxph(Surv(time, status) ~ sex + diagnosis + loc + ki + gtv + stereo, data = BrainCancer)
fit

# Comment: diagnosisHG and ki are highly significant.

# d. Stratify the data by the value of ki. (Since only one observation has ki=40, 
# you can group that observation together with the observations that have ki=60.)
# Plot Kaplan-Meier survival curves for each of the five strata, adjusted for
# the other predictors.

library(ggfortify)

modaldata <- data.frame(
     sex = rep("Female", 5),
     diagnosis = rep("Meningioma", 5),
     loc = rep("Supratentorial", 5),
     ki = c(60, 70, 80, 90, 100),
     gtv = rep(mean(BrainCancer$gtv), 5),
     stereo = rep("SRT", 5)
)

survplots <- survfit(fit, newdata = modaldata)
plot(survplots, xlab = "Months", ylab = "Survival Probability", col = 2:6)
legend("bottomleft", c("60", "70", "80", "90", "100"), col = 2:6, lty = 1)
