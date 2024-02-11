# ISLP Chapter 11 - Survival Analysis and Censored Data
# Applied exercise 10

# import librarires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ISLP.models import ModelSpec as MS
from ISLP import load_data

import warnings
warnings.filterwarnings('ignore')

# a.
# Using the brain cancer dataset, Plot the Kaplan-Meier survival curve with
# ±1 standard error bands, using the KaplanMeierFitter() estimator in the 
# lifelines package.

# load dataset
bc_df = load_data('BrainCancer')
bc_df

bc_df.info()

# count 'status' 
bc_df['status'].value_counts()

# 'status' is coded: 1 = uncensored observation (event was achieved during the time period)
# 0 = censored (survived event)

# Create KaplanMeierFitter object
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()

# Fit the Kaplan-Meier estimator to the data (alpha = 0.33)
kmf.fit(durations = bc_df['time'], event_observed = bc_df['status'], alpha = 0.33)

# Plot the Kaplan-Meier survival curve with ±1 standard error bands
fig, ax =  plt.subplots(figsize=(10,10))
kmf.plot_survival_function(color = 'C0', at_risk_counts = True, ax = ax)
ax.set(
    title='Kaplan-Meier survival curve',
    xlabel='Months',
    ylabel='Estimated Probability of Survival'
)

# b.
from scipy.stats import bootstrap

# define variables
time_column = 'time'
event_column = 'status'

# Set the number of bootstrap samples and size of each sample
B = 200
n_bootstrap = 88

# Function to compute Kaplan-Meier survival curve for a given bootstrap sample
def compute_kaplan_meier(data):
    kmf.fit(durations=data[time_column], event_observed=data[event_column], alpha = 0.33)
    return kmf.survival_function_

compute_kaplan_meier(bc_df)

km_df = compute_kaplan_meier(bc_df).reset_index()

# Perform bootstrap sampling and calculate confidence intervals
bootstrap_dict = {}

for i in range(B):
    # Generate a bootstrap sample by sampling with replacement
    bootstrap_sample = np.random.choice(km_df['KM_estimate'], size=len(bc_df), replace=True)
    bootstrap_sample[::-1].sort()

    # Apply the function to the bootstrap sample
    array_name = f'Column{i}'
    bootstrap_dict[array_name] = bootstrap_sample

bootstrap_df = pd.DataFrame(bootstrap_dict)

# calculate standard deviation for each row
bootstrap_mean = bootstrap_df.mean(axis=1, numeric_only=True)
bootstrap_std = bootstrap_df.std(axis=1, numeric_only=True)

bs_df = pd.concat([km_df['timeline'], bootstrap_mean, bootstrap_std], axis=1) \
    .set_axis(['time', 'bs_mean', 'bs_std'], axis=1) \
    .dropna() \
    .assign(bs_plus_one_se = lambda x: x['bs_mean'] + x['bs_std']) \
    .assign(bs_minus_one_se = lambda x: x['bs_mean'] - x['bs_std'])

bs_df

# convert kmf.survival_function_.values to pandas Series
my_list = map(lambda x: x[0], kmf.survival_function_.values)
kmf_values_series = pd.Series(my_list)

# Plot the Kaplan-Meier survival curve with ±1 standard error bands
plt.plot(kmf.timeline, kmf.survival_function_.values, label='Original', color='blue')
plt.fill_between(bs_df['time'], kmf_values_series - bs_df['bs_std'], 
                 kmf_values_series + bs_df['bs_std'], 
                 alpha=0.2, color='blue')


# c.
# Fit a Cox proportional hazards model that uses all of the predictors to 
# predict survival. Summarize the main findings.
from lifelines import CoxPHFitter

cleaned = bc_df.dropna()
all_MS = MS(cleaned.columns, intercept=False)
all_df = all_MS.fit_transform(cleaned)

coxph = CoxPHFitter
fit_all = coxph().fit(all_df,
                      'time',
                      'status')

fit_all.print_summary()

fit_all.summary[['coef', 'se(coef)', 'p']]

# Summary
# 1. Diagnosis factors and ki are statistically significant to the patient
# survival probability.
# 
# 2. The risk associated with HG glioma is more than eight times 
# (i.e. e^{2.15}=8.62) the risk associated with meningioma.
# 
# 3. In addition, larger values of the Karnofsky index, `ki`, are associated 
# with lower risk, i.e. longer survival.

# d.
# Stratify the data by the value of ki.

cleaned['ki'].value_counts()

cleaned_ki = cleaned.copy()
cleaned_ki['ki'] = np.where(cleaned_ki['ki'] == 40, 60, cleaned_ki['ki'])

cleaned_ki['ki'] = cleaned_ki['ki'].astype('category')

# create modal data
levels = cleaned_ki['ki'].unique()

def representative(series):
    if hasattr(series.dtype, 'categories'):
        return pd.Series.mode(series)
    else:
        return series.mean()
    
modal_data = cleaned_ki.apply(representative, axis=0)

modal_df = pd.DataFrame(
              [modal_data.iloc[0] for _ in range(len(levels))])

modal_df['ki'] = levels
modal_df

# construct the model matrix
modal_X = all_MS.transform(modal_df)
modal_X.index = levels
modal_X

# use 'predict_survival_function()' to obtain the estimated survival function
predicted_survival = fit_all.predict_survival_function(modal_X)
predicted_survival

# survival plots
fig, ax = plt.subplots(figsize=(10, 10))
predicted_survival.plot(ax=ax)

# The KM plots by ki confirm rou conclusion using the COX PH regression model.
# An increase in ki results in a higher survival propbability, adjusting for the
# predictors.