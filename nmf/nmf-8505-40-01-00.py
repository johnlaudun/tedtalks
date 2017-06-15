#! /usr/bin/env python

import pandas
import re
import sklearn.feature_extraction.text as sktext 
from sklearn.decomposition import NMF
import numpy as np


# Create pandas dataframe & lists
colnames = ['author', 'title', 'date' , 'length', 'text']
df = pandas.read_csv('../data/talks_3a.csv', names=colnames)
talks = df.text.tolist()

# Import stoplist
stopwords = re.split('\s+', open('../data/stopwords_2.txt', 'r').read().lower())

# TFIDF parameters
n_top_words = 15
max_percent = 0.85
min_percent = 0.05 # One percent = 20 talks (so not enought to warrant a topic?)

# Create TFIDF matrix
vectorizer = sktext.TfidfVectorizer(lowercase = True, 
                             stop_words = stopwords,
                             max_df = max_percent,
                             min_df = min_percent)
td_matrix = vectorizer.fit_transform(talks)
print(td_matrix.shape)


# Fit NMF
model = NMF(n_components = 40,
            solver='cd',
            random_state = 1,
            alpha = 0.1,
            l1_ratio = 0).fit(td_matrix)

W = model.fit_transform(td_matrix)
H = model.components_
print(W.shape, H.shape)


np.savetxt("../nmf/8505-40-1-01-00-dtm.csv", H, delimiter=",", fmt = "%s")
np.savetxt("../nmf/8505-40-1-01-00-twm.csv", W, delimiter=",", fmt = "%s")


def print_top_words(model, feature_names, n_top_words):
    for topic_id, topic in enumerate(model.components_):
        print('\nTopic {}:'.format(int(topic_id)))
        print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
              +', ' for i in topic.argsort()[:-n_top_words - 1:-1]]))

features = vectorizer.get_feature_names()

print("Topics in NMF model:")
print_top_words(model, features, n_top_words) #n_top_words can be changed on the fly