# Bayesian Survival Analysis
# Source: https://modernstatisticswithr.com/regression.html#survival

# load packages
library(survival)
library(rstanarm)
library(bayesplot)
library(ISLR2)

# setup parallel processing
options(mc.cores = parallel::detectCores())

# load datset
data("BrainCancer")
BrainCancer

# Fit proportional hazards model using cubic M-splines (similar
# but not identical to the Cox model!):
mod_sex <- stan_surv(Surv(time, status) ~ sex, data = BrainCancer)
summary(mod_sex)

# fit PH model adding stereo features
mod_stereo <- stan_surv(Surv(time, status) ~ sex + stereo, data = BrainCancer)
summary(mod_stereo)

# extract posterior draws
posterior <- as.array(mod_stereo)
dim(posterior)

# plot posterior uncertainty intervals
color_scheme_set("red")
mcmc_intervals(posterior, pars = vars(sexMale, stereoSRT))

# posterior parameters distribution areas
mcmc_areas(
     posterior, 
     pars = vars(sexMale, stereoSRT), 
     prob = 0.8, # 80% intervals
     prob_outer = 0.95, # 95%
     point_est = "mean"
)
