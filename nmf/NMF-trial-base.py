#! /usr/bin/env python 

# =-=-=-=-=-=
# Read CSV into DataFrame and then create lists
# =-=-=-=-=-=

import pandas
import re

# Create pandas dataframe & lists
colnames = ['author', 'title', 'date' , 'length', 'text']
df = pandas.read_csv('../data/talks_3a.csv', names=colnames)
talks = df.text.tolist()


# =-=-=-=-=-=-=-=-=-=-=
# Create citations to identify individual texts
# =-=-=-=-=-=-=-=-=-=-= 

# authors = df.author.tolist()
# dates = df.date.tolist()
# years = [re.sub('[A-Za-z ]', '', item) for item in dates]
# authordate = [author+" "+year for author, year in zip(authors, years)]


# In[25]:

# =-=-=-=-=-=
# Clean and Tokenize, then Drop Stopwords
# =-=-=-=-=-=

from nltk.tokenize import WhitespaceTokenizer

# From the Stopwords Notebook:
tokenizer = WhitespaceTokenizer()
# stopwords = re.split('\s+', open('../data/tt_stop.txt', 'r').read().lower())

# Loop to tokenize, stop, and stem (if needed) texts.
texts = []
for talk in talks:   
    # clean and tokenize document string
    raw = re.sub(r"[^\w\d'\s]+",'', talk).lower()
    tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
    # stopped_tokens = [i for i in tokens if not i in stopwords]
    # stem tokens
    # stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    # add tokens to list
    texts.append(tokens)

# =-=-=-=-=-=-=-=-=-=-=
# Re-Assemble Texts as Strings from Lists of Words
# =-=-=-=-=-=-=-=-=-=-= 

strungs = []
for text in texts:
    strung = ' '.join(text)
    strungs.append(strung)


# In[28]:

# =-=-=-=-=-=
# Get NMF topics
# =-=-=-=-=-=

import sklearn.feature_extraction.text as sk_text
from sklearn.decomposition import NMF

# All our variables are here to make it easier to make adjustments
n_samples = len(strungs)
n_features = 3000
n_topics = 40
n_top_words = 15
max_percent = 0.85
min_percent = 0.05
tt_stopwords = open('../data/stopwords-tt-20170523.txt', 'r').read().splitlines()

# Get tf-idf features for NMF
vectorizer = sk_text.TfidfVectorizer(max_df = max_percent, 
                                     min_df = min_percent,
                                     max_features = n_features,
                                     stop_words = tt_stopwords)
tfidf = vectorizer.fit_transform(strungs)

# Fit the NMF model
nmf = NMF(n_components = n_topics,
          random_state = 1,
          alpha = 0.1,
          l1_ratio = 0.5).fit(tfidf)
print("Fitting the NMF model with {} topics for {} documents with {} features."
      .format(n_topics, n_samples, n_features))


# In[29]:

# =-=-=-=-=-=
# Get NMF printing
# =-=-=-=-=-=

def print_top_words(model, feature_names, n_top_words):
    for topic_id, topic in enumerate(model.components_):
        print('\nTopic {}:'.format(int(topic_id)))
        print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
              +', ' for i in topic.argsort()[:-n_top_words - 1:-1]]))

features = vectorizer.get_feature_names()
# print(features)

# This should work now.

print("Topics in NMF model:")
print_top_words(nmf, features, n_top_words) #n_top_words can be changed on the fly


# In[30]:

dtm = tfidf.toarray()
doctopic = nmf.fit_transform(dtm) # This is an array

# features = vectorizer.get_feature_names() # This is already done above.

# =-=-=-=-=-=-=-=-=-=-=
# Unused functionality
# =-=-=-=-=-=-=-=-=-=-= 
# print(features)
# for topicidx in enumerate(nmf.components_):
#     print(topicidx)


# In[32]:

# =-=-=-=-=-=
# Saving output to CSV
# =-=-=-=-=-=

import numpy as np

# Since DOCTOPIC is an array, you can just do:
#      np.savetxt("foo.csv", doctopic, delimiter=",", fmt = "%s")
# http://stackoverflow.com/questions/6081008/dump-a-numpy-array-into-a-csv-file
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

