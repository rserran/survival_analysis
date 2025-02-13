# ISLP Chapter 11 - Survival Analysis and Censored Data
# Applied exercise 10 (b)

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ISLP import load_data
from lifelines import KaplanMeierFitter

import warnings
warnings.filterwarnings('ignore')

# load BrainCancer dataset
data = load_data("BrainCancer")
data

# Extract survival time (y) and event indicator (delta)
y = data["time"]  # Survival time
delta = data["status"]  # Event indicator (1=event, 0=censored)

# define parameter variables
n = len(y)  # Number of observations
B = 200  # Number of bootstrap samples
kmf = KaplanMeierFitter()

# Create an array to store survival probabilities at each timepoint across bootstrap samples
unique_times = np.sort(np.unique(y))
survival_curves = np.zeros((B, len(unique_times)))

# Bootstrap resampling
for b in range(B):
    
    indices = np.random.choice(np.arange(n), size=n, replace=True)
    y_bootstrap = y.iloc[indices].values
    delta_bootstrap = delta.iloc[indices].values
    
    # Fit Kaplan-Meier estimator
    kmf.fit(y_bootstrap, event_observed=delta_bootstrap)
    
    # Interpolate survival probabilities at unique times
    survival_curves[b, :] = np.interp(unique_times, kmf.survival_function_.index, kmf.survival_function_['KM_estimate'])

# Compute standard error at each time point
se_km = np.std(survival_curves, axis=0, ddof=1)

# Store results in a DataFrame
results = pd.DataFrame({
    "time": unique_times,
    "se": se_km
})

print(results)

# save results to csv (for future use)
results.to_csv('./km_standard_errors.csv', index=False)

# Fit the Kaplan-Meier estimator on the original data
kmf.fit(y, event_observed=delta)

# Interpolate the Kaplan-Meier estimates to match the unique times
km_estimates = np.interp(unique_times, kmf.survival_function_.index, kmf.survival_function_['KM_estimate'])

# Plot the Kaplan-Meier survival curve with bootstrap standard error
plt.figure(figsize=(8, 6))
plt.step(kmf.survival_function_.index, kmf.survival_function_['KM_estimate'], where="post", label="Kaplan-Meier Estimate", color='blue')
plt.fill_between(unique_times, km_estimates - results["se"], km_estimates + results["se"], color='blue', alpha=0.2, label="Bootstrap SE")

plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.title("Kaplan-Meier Survival Curve with Bootstrap SE")
plt.legend()
plt.grid()
plt.show()