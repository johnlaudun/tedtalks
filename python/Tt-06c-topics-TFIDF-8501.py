
# =-=-=-=-=-=
# Read CSV into DataFrame and then create lists
# =-=-=-=-=-=

import pandas
import re

# Create pandas dataframe & lists
colnames = ['author', 'title', 'date' , 'length', 'text']
df = pandas.read_csv('../data/talks_3a.csv', names=colnames)
talks = df.text.tolist()

# We are not going to need the identifiers for this run, so I'm leaving them commented out.
# =-=-=-=-=-=-=-=-=-=-=
# Create citations to identify individual texts
# =-=-=-=-=-=-=-=-=-=-= 

# authors = df.author.tolist()
# dates = df.date.tolist()
# years = [re.sub('[A-Za-z ]', '', item) for item in dates]
# authordate = [author+" "+year for author, year in zip(authors, years)]


import sklearn.feature_extraction.text as sktext 
from sklearn.decomposition import NMF
import numpy as np

# Import stoplist
stopwords = re.split('\s+', open('../data/stopwords_2.txt', 'r').read().lower())

# TFIDF parameters
max_percent = 0.85
min_percent = 0.01 # One percent = 20 talks (so not enought to warrant a topic?)

# Create TFIDF matrix
vectorizer = sktext.TfidfVectorizer(lowercase = True, 
                             stop_words = stopwords,
                             max_df = max_percent,
                             min_df = min_percent)
td_matrix = vectorizer.fit_transform(talks)
print(td_matrix.shape)