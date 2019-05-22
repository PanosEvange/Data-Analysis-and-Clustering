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
# %matplotlib inline
# endregion

# ## __Read data__

# region
initialDataFrame = pd.read_csv('../data/crime.csv', engine='python')

initialDataFrame # printToBeRemoved
# endregion

# ## __Do some data preparation__

# region
# drop the columns that we don't need
processedDataFrame = initialDataFrame.drop(['Location'], axis=1)

# replace some NaN values
processedDataFrame = processedDataFrame.fillna({'SHOOTING':"Ν"})

processedDataFrame['SHOOTING'].unique()  # printToBeRemoved
# endregion

processedDataFrame  # printToBeRemoved

# ## __Data Research__

# 1. #### Multitude of crimes by year, by month, by date and by district

# region
# groupBy year
yearMultitudeDf = processedDataFrame.groupby(['YEAR']).count()

# replace month numbers to month names
monthMultitudeDf = processedDataFrame.copy()
monthMultitudeDf['MONTH'] = monthMultitudeDf['MONTH'].apply(lambda x: calendar.month_abbr[x])

# groupBy month
monthMultitudeDf = monthMultitudeDf.groupby(['MONTH']).count()

# groupBy day
dayMultitudeDf = processedDataFrame.groupby(['DAY_OF_WEEK']).count()

# groupBy district
districtMultitudeDf = processedDataFrame.groupby(['DISTRICT']).count()

print("Count by year:")  # printToBeRemoved
print(yearMultitudeDf['INCIDENT_NUMBER'])  # printToBeRemoved
print("--------------")  # printToBeRemoved

print("Count by month:")  # printToBeRemoved
print(monthMultitudeDf['INCIDENT_NUMBER'])  # printToBeRemoved
print("--------------")  # printToBeRemoved

print("Count by day:")  # printToBeRemoved
print(dayMultitudeDf['INCIDENT_NUMBER'])  # printToBeRemoved
print("--------------")  # printToBeRemoved

print("Count by district:")  # printToBeRemoved
print(districtMultitudeDf['INCIDENT_NUMBER'])  # printToBeRemoved
print("--------------")  # printToBeRemoved
# endregion


