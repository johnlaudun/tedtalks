#! /usr/bin/env python 

import pandas
import re
import sklearn.feature_extraction.text as sktext 


# Create pandas dataframe & lists
colnames = ['author', 'title', 'date' , 'length', 'text']
df = pandas.read_csv('../data/talks_3a.csv', names=colnames)
talks = df.text.tolist()

# Import stoplist
stopwords = re.split('\s+', open('../data/stopwords_2.txt', 'r').read().lower())

# Create TFIDF matrix
vectorizer = sktext.TfidfVectorizer(lowercase = True, 
                             stop_words = stopwords,
                             max_df = 0.85,
                             min_df = 0.05)
td_matrix = vectorizer.fit_transform(talks)

# Save features (words) to file
with open('../nmf/features-tfidf_only.txt', 'w') as the_file:
    for item in vectorizer.get_feature_names():
        the_file.write("%s\n" % item)