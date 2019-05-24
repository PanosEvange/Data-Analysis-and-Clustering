# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_markers: region,endregion
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # <center>Data Mining Project 2 Spring semester 2018-2019</center>
# ## <center>Παναγιώτης Ευαγγελίου &emsp; 1115201500039</center>
# ## <center>Ευάγγελος Σπίθας &emsp;&emsp;&emsp;&ensp; 1115201500147</center>

# ___

# ## Do all the necessary imports for this notebook

# region
import numpy as np
import pandas as pd
import calendar

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# endregion

# %matplotlib inline

# ## __Read data__

# region
initialDataFrame = pd.read_csv('../data/crime.csv', engine='python')

# initialDataFrame # printToBeRemoved
# endregion

# ## __Do some data preparation__

# region
# drop the columns that we don't need
processedDataFrame = initialDataFrame.drop(['Location'], axis=1)

# replace some NaN values
processedDataFrame = processedDataFrame.fillna({'SHOOTING': "N"})

processedDataFrame['SHOOTING'].unique()  # printToBeRemoved
# endregion

# ## __Data Research__

# 1. #### Count of crimes per year, per month, per date and per district

# region
# groupBy year
# yearCountDf = processedDataFrame.groupby(['YEAR']).count() # to be removed
yearCountSeries = processedDataFrame.groupby(['YEAR'])['INCIDENT_NUMBER'].count()

# replace month numbers to month names
monthCountDf = processedDataFrame.copy()
monthCountDf['MONTH'] = monthCountDf['MONTH'].apply(lambda x: calendar.month_abbr[x])

# groupBy month
monthCountSeries = monthCountDf.groupby(['MONTH'])['INCIDENT_NUMBER'].count()
# monthCountDf = monthCountDf.groupby(['MONTH']).count() # to be removed

# groupBy day
# dayCountDf = processedDataFrame.groupby(['DAY_OF_WEEK']).count() # to be removed
dayCountSeries = processedDataFrame.groupby(['DAY_OF_WEEK'])['INCIDENT_NUMBER'].count()

# groupBy district
# districtCountDf = processedDataFrame.groupby(['DISTRICT']).count() # to be removed
districtCountSeries = processedDataFrame.groupby(['DISTRICT'])['INCIDENT_NUMBER'].count()

print("Count by year:")  # printToBeRemoved
print(yearCountSeries)  # printToBeRemoved
print("--------------")  # printToBeRemoved

print("Count by month:")  # printToBeRemoved
print(monthCountSeries)  # printToBeRemoved
print("--------------")  # printToBeRemoved

print("Count by day:")  # printToBeRemoved
print(dayCountSeries)  # printToBeRemoved
print("--------------")  # printToBeRemoved

print("Count by district:")  # printToBeRemoved
print(districtCountSeries)  # printToBeRemoved
print("--------------")  # printToBeRemoved

# for distr, count in districtCountSeries.items():
#     print("District: ", distr, " has ", count, " crimes!")
# endregion

# 2. #### Find the maximum count of shootings by year and by district

# region
# replace Y -> 1, N -> 0 so as we can sum the shootings
shootingDataFrame = processedDataFrame.copy()
shootingDataFrame['SHOOTING'] = shootingDataFrame['SHOOTING'].map(dict(Y=1, N=0))

# groupBy year
yearShootings = shootingDataFrame.groupby(['YEAR'])['SHOOTING'].sum()

# groupBy district
districtShootings = shootingDataFrame.groupby(['DISTRICT'])['SHOOTING'].sum()

print("Shootings by year:")  # printToBeRemoved
print(yearShootings)  # printToBeRemoved
print("Max year is:")  # printToBeRemoved
print(yearShootings[yearShootings == yearShootings.max()])  # printToBeRemoved
print("--------------")  # printToBeRemoved
print("Shootings by district:")  # printToBeRemoved
print(districtShootings)  # printToBeRemoved
print("Max district is:")  # printToBeRemoved
print(districtShootings[districtShootings == districtShootings.max()])  # printToBeRemoved
print("--------------")  # printToBeRemoved
# endregion

# 3. #### Check if crimes are more during the day than during the night

# region
# make new column that represents day or night
dayNightDataFrame = processedDataFrame.copy()
dayNightDataFrame['DAY_NIGHT'] = np.where(((dayNightDataFrame['HOUR'] >= 7) & (dayNightDataFrame['HOUR'] <= 17)),
                                          'day', 'night')

dayCrimesCount = (dayNightDataFrame['DAY_NIGHT'] == 'day').sum()
nightCrimesCount = (dayNightDataFrame['DAY_NIGHT'] == 'night').sum()

print("Count of crimes during day is:")  # printToBeRemoved
print(dayCrimesCount)  # printToBeRemoved

print("Count of crimes during night is:")  # printToBeRemoved
print(nightCrimesCount)  # printToBeRemoved

# dayNightDataFrame  # printToBeRemoved
# endregion

# 4. #### Find the most common type of crime that is committed during the day

# region
# make a dataFrame that is consisted only of crimes that are committed during the day
onlyDayCrimes = dayNightDataFrame.copy()
onlyDayCrimes = onlyDayCrimes[(onlyDayCrimes['DAY_NIGHT'] == "day")]

# if we need only the name of the code then we can do the following
mostCommonOffenseCode = onlyDayCrimes['OFFENSE_CODE_GROUP'].mode()

# if we want to know the count of crimes of this type then
# groupBy by offense_code
codeCount = onlyDayCrimes.groupby(['OFFENSE_CODE_GROUP'])['INCIDENT_NUMBER'].count()

print("the most common type of crime that is committed during the day is ", mostCommonOffenseCode)
print(codeCount[codeCount == codeCount.max()])
# onlyDayCrimes  # printToBeRemoved
# endregion

# 5. #### Clustering based on location

# Let's try a scatter plot with seaborn first

# region
locationDf = processedDataFrame[['Lat', 'Long']]

# remove missing values
locationDf = locationDf.dropna()

specificLocation = locationDf.loc[(locationDf['Lat'] > 40) & (locationDf['Long'] < -60)]

ax = sns.scatterplot(x="Long", y="Lat", data=specificLocation)
# endregion

# Let's try matplotlib also

# region
x = specificLocation['Long']
y = specificLocation['Lat']

colors = np.random.rand(len(specificLocation))

plt.figure(figsize=(12, 12))
plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()
# endregion
