#! /usr/bin/env python 

import pandas
import re
from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create pandas dataframe & lists
colnames = ['author', 'title', 'date' , 'length', 'text']
df = pandas.read_csv('../data/talks_3a.csv', names=colnames)
talks = df.text.tolist()

# Import stoplist
stopwords = re.split('\s+', open('../data/stopwords_2.txt', 'r').read().lower())

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
    

# Create TFIDF matrix
vectorizer = TfidfVectorizer(max_df = 0.85, 
                             min_df = 0.05)
td_matrix = vectorizer.fit_transform(strings)

# Create a file
with open('../nmf/features-nltk_tfidf.txt', 'w') as the_file:
    for item in vectorizer.get_feature_names():
        the_file.write("%s\n" % item)