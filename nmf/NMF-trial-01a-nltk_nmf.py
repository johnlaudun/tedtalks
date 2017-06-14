#! /usr/bin/env python 


# =-=-=-=-=-=-=-=-=-=-=
# Imports
# =-=-=-=-=-=-=-=-=-=-= 

import pandas
import re
from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np


# =-=-=-=-=-=-=-=-=-=-=
# Inputs
# =-=-=-=-=-=-=-=-=-=-= 

# Create pandas dataframe & lists
colnames = ['author', 'title', 'date' , 'length', 'text']
df = pandas.read_csv('../data/talks_3a.csv', names=colnames)
talks = df.text.tolist()

# Import stoplist
stopwords = re.split('\s+', open('../data/stopwords_2.txt', 'r').read().lower())


# =-=-=-=-=-=
# Drop Stopwords
# =-=-=-=-=-=

tokenizer = WhitespaceTokenizer()

# Loop to tokenize, stop, and stem (if needed) texts.
texts = []
for talk in talks:   
    # clean and tokenize document string
    raw = re.sub(r"[^\w\d'\s]+",'', talk).lower()
    tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
    passed = [i for i in tokens if not i in stopwords]
    # add tokens to list
    texts.append(passed)

# Re-Assemble Texts as Strings from Lists of Words
strings = []
for text in texts:
    the_string = ' '.join(text)
    strings.append(the_string)


# =-=-=-=-=-=
# NMF Topics
# =-=-=-=-=-=


# TFIDF parameters
n_samples = len(strings)
n_features = 3000
n_top_words = 15
max_percent = 0.85
min_percent = 0.05

# Create TFIDF matrix
vectorizer = TfidfVectorizer(max_df = max_percent, 
                             min_df = min_percent,
                             max_features = n_features)
tfidf = vectorizer.fit_transform(strings)

# Fit the NMF model
model = NMF(n_components = 40,
              random_state = 1,
              alpha = 0.1,
              l1_ratio = 0.5).fit(tfidf)

# =-=-=-=-=-=
# NMF printing
# =-=-=-=-=-=

def print_top_words(model, feature_names, n_top_words):
    for topic_id, topic in enumerate(model.components_):
        print('\nTopic {}:'.format(int(topic_id)))
        print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
              +', ' for i in topic.argsort()[:-n_top_words - 1:-1]]))

features = vectorizer.get_feature_names()

print("Topics in NMF model:")
print_top_words(nmf, features, n_top_words) #n_top_words can be changed on the fly



# =-=-=-=-=-=
# Saving output to CSV
# =-=-=-=-=-=

W = model.fit_transform(tfidf) # This is the topic-word matrix
H = model.components_  # This is the doc-topic matrix



# Since DOCTOPIC is an array, you can just do:
#      np.savetxt("foo.csv", doctopic, delimiter=",", fmt = "%s")
#
# The above won't give you the names of the files. Instead try this:

topsnum = np.array([list(range(n_topics))])
# topsnum = np.indices((1,n_topics))[1] <-- this is more than we need,
#                                           but it's cool to know more tricks
#
# Two ways to get an array that is of the form [[0,1,2,3,...]].
# It will have the desired dimensions of (1,35) which is what we want

fileheader = np.concatenate((np.array([["citations"]]), topsnum),axis = 1)
authordate = np.array([df.author])

docTopics = np.concatenate((authordate.T, doctopic), axis = 1)
docTopics = np.concatenate((fileheader, docTopics), axis = 0)

np.savetxt("../outputs/nmf_topics_20170523.csv", doctopic, delimiter=",", fmt = "%s")
#np.savetxt("../data/nmf_topics.csv", docTopics, delimiter=",", fmt = "%s")

