# https://www.pythonfordatascience.org/
# https://thedatascientist.com/how-to-do-a-t-test-in-python/


import numpy as np
from scipy import stats 
from numpy.random import seed
from numpy.random import randn
from numpy.random import normal
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt # plots
import seaborn as sns
sns.set_theme()

# assigning random state seed to 1

seed(1)
sample = normal(150,10,20) # mean, sd, size
print('Sample: ', sample)

t_stat, p_value = ttest_1samp(sample, popmean=155)
print("T-statistic value: ", t_stat)  
print("P-Value: ", p_value)


# seed the random number generator
seed = (1) 
# create two independent sample groups
sample1 = normal(30, 16, 50)
sample2 = normal(33, 18, 50)
print('Sample 1: ',sample1)
print('Sample 2: ',sample2)

t_stat, p_value = ttest_ind(sample1, sample2)
print("T-statistic value: ", t_stat)  
print("P-Value: ", p_value)


# seed the random number generator
seed = (1)
# create two dependent sample groups
sample1 = normal(30, 16, 50)
sample2 = normal(33, 18, 50)
print('Sample 1: ', sample1)
print('Sample 2: ', sample2)


t_stat, p_value = ttest_rel(sample1, sample2)
print("T-statistic value: ", t_stat)  
print("P-Value: ", p_value)





df = pd.read_csv("https://raw.githubusercontent.com/researchpy/Data-sets/master/blood_pressure.csv")
df.info()

stats.ttest_ind(df['bp_after'][df['sex'] == 'Male'],
                df['bp_after'][df['sex'] == 'Female'])

sampling_difference = df['bp_after'][df['sex'] == 'Male'].values - \
                      df['bp_after'][df['sex'] == 'Female'].values

stats.shapiro(sampling_difference)
fig = plt.figure(figsize= (20, 10))
ax = fig.add_subplot(111)

normality_plot, stat = stats.probplot(sampling_difference, plot= plt, rvalue= True)
ax.set_title("Probability plot of sampling difference", fontsize= 20)
ax.set

plt.show()

stats.levene(df['bp_after'][df['sex'] == 'Male'],
             df['bp_after'][df['sex'] == 'Female'],
             center= 'mean')

fig = plt.figure(figsize= (20, 10))
ax = fig.add_subplot(111)


p_bp_male = plt.hist(df['bp_after'][df['sex'] == 'Male'], label= "Male",
                     density= True,
                     alpha=0.75)
p_bp_female = plt.hist(df['bp_after'][df['sex'] == 'Female'], label= "Female",
                       density= True,
                       alpha=0.75)


plt.suptitle("Distribution of Post-Blood Pressure \n between Males and Females", fontsize= 20)
plt.xlabel("Blood Pressure", fontsize= 16)
plt.ylabel("Probability density", fontsize= 16)

plt.text(133, .025,
         f"$\mu= {df['bp_after'][df['sex'] == 'Female'].mean(): .1f}, \ \sigma= {df['bp_after'][df['sex'] == 'Female'].std(): .1f}$")
plt.text(160, .025,
         f"$\mu= {df['bp_after'][df['sex'] == 'Male'].mean(): .1f}, \ \sigma= {df['bp_after'][df['sex'] == 'Male'].std(): .1f}$")


plt.show()