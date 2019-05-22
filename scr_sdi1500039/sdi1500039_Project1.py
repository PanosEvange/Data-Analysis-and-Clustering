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

# ### Do all the necessary imports for this notebook

# region
import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
# endregion

# ### Read data

# region
initialDataFrame = pd.read_csv('../data/crime.csv', engine='python')

initialDataFrame # printToBeRemoved
# endregion

# ### Do some data preparation

# region
# drop the columns that we don't need
processedDataFrame = initialDataFrame.drop(['Location'], axis=1)

# replace some NaN values
processedDataFrame = processedDataFrame.fillna({'SHOOTING':"Ν"})

processedDataFrame['SHOOTING'].unique() # printToBeRemoved
# endregion

processedDataFrame