# Hedonometer test code
import sys, os
from labMTsimple.storyLab import *
import codecs ## handle utf8

import pandas as pd
import numpy as np

# 0 - Break the talks into separate data files. 
# 0a - Import and sort the data file by NUMDATE

all_talks = pd.read_csv('./talks_6d.csv')
just_text = all_talks.loc[:,"text"]
just_dates = all_talks.loc[:,"numDate"]

dates_text = pd.concat([just_dates,just_text], axis = 1)
dt_sort = dates_text.sort_values("numDate", kind="mergesort")

num_talks = dt_sort.shape[0]

# 0b - For range of dates check which dates have talks
#    - If there are talks add to the list of allowable dates
curr_date = dt_sort.loc[:,"numDate"].min()
last_date = dt_sort.loc[:,"numDate"].max()

for i in range(num_talks):
	

# 0c - Dump all the text into one file per month

# 0d - Create a file that is this month and all of the previous talks

# 1 - Compute word shift graphs
#   - Loop over all the allowable dates




# Pages consulted - 
# https://chrisalbon.com/python/pandas_dataframe_importing_csv.html
# http://pythonhow.com/accessing-dataframe-columns-rows-and-cells/
# https://pandas.pydata.org/pandas-docs/stable/merging.html
# https://stackoverflow.com/questions/493819/python-join-why-is-it-string-joinlist-instead-of-list-joinstring