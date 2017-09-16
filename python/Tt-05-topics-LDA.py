

# =-=-=-=-=-=
# Read CSV into DataFrame and then create lists
# =-=-=-=-=-=

import pandas
import re


# Create pandas dataframe
colnames = ['author', 'title', 'date' , 'length', 'text']
df = pandas.read_csv('../data/talks_3a.csv', names=colnames)

# Create lists for the data
talks = df.text.tolist()
authors = df.author.tolist()
dates = df.date.tolist()

# Getting only the years from dates list
years = [re.sub('[A-Za-z ]', '', item) for item in dates]

# Combining year with presenter for citation
authordate = [author+" "+year for author, year in zip(authors, years)]

# Just to check to see if things are synced,
# let's create a new df with the two lists.

citations = pandas.DataFrame(
    {'citation': authordate,
     'text': talks,
    })

# Uncomment to show that the citation and the text are paired correctly:
# citations.head()



# =-=-=-=-=-=-=-=-=-=-=
# Settings & Display Functions
# =-=-=-=-=-=-=-=-=-=-= 

n_topics = 50
n_features = 5000
n_top_words = 10
n_top_documents = 3


stopwords = re.split('\s+', open('../data/stopwords_all.txt', 'r').read().lower())

def display_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("| "+str(topic_idx)+" |"+' '.join([feature_names[i] + ' ' + str(round(topic[i], 2))
              +',' for i in topic.argsort()[:-n_top_words - 1:-1]])+"|")
        
# Both NMF **and** LDA produce two matrices: 
# H - words to topics
# W - topics to documents

def topics_and_docs(H, W, feature_names, documents, n_top_words, n_top_documents):
    for topic_idx, topic in enumerate(H):
        print("Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:n_top_documents]
        for doc_index in top_doc_indices:
            print(documents[doc_index])


# =-=-=-=-=-=
# Generate LDA Model
# =-=-=-=-=-=

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# LDA can only use raw term counts (virtual BoW)
tf_vectorizer = CountVectorizer(max_df = 0.95, 
                                min_df = 2, 
                                max_features = n_features, 
                                stop_words = stopwords)
tf = tf_vectorizer.fit_transform(talks)
tf_feature_names = tf_vectorizer.get_feature_names()

lda = LatentDirichletAllocation(n_topics = n_topics, 
                                max_iter = 5, 
                                learning_method = 'online', 
                                learning_offset = 50.,
                                random_state = 0).fit(tf)
lda_W = lda_model.transform(tf)
lda_H = lda_model.components_

# Display topics as word cluster
display_topics(lda, tf_feature_names, n_top_words)


topics_and_docs(lda_H, lda_W, tf_feature_names, talks, n_top_words, n_top_documents)