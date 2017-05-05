# =-=-=-=-=-=-=-=-=-=-=
# Data Load & Tokenize
# =-=-=-=-=-=-=-=-=-=-= 

import pandas
import re
from nltk.tokenize import WhitespaceTokenizer

# LOAD csv into dataframe
colnames = ['author', 'title', 'date' , 'length', 'text']
df = pandas.read_csv('../data/talks_3.csv', names=colnames)
talks = df.text.tolist()

# All this for labels
authors = df.author.tolist()
dates = df.date.tolist()
years = [re.sub('[A-Za-z ]', '', item) for item in dates]
authordate = [author+" "+year for author, year in zip(authors, years)]

# TOKENIZE
tokenizer = WhitespaceTokenizer()
texts = []
for talk in talks:   
    raw = re.sub(r"[^\w\d'\s]+",'', talk).lower()
    tokens = tokenizer.tokenize(raw)
    texts.append(tokens)


# =-=-=-=-=-=-=-=-=-=-=
# Small Test Corpus
# =-=-=-=-=-=-=-=-=-=-= 

test = texts[0:5]


# =-=-=-=-=-=-=-=-=-=-=
# Function to collect word positions within a text (as a word list)
# =-=-=-=-=-=-=-=-=-=-= 

def word_positions(listname):
    from collections import defaultdict
    words_with_positions = defaultdict(list)
    for position, word in enumerate(listname):
        words_with_positions[word].append(float(position)/len(listname))
    return(words_with_positions)

# word_positions(texts[0]) # Check output


# =-=-=-=-=-=-=-=-=-=-=
# Develop "super dictionary" of all the texts involved
# =-=-=-=-=-=-=-=-=-=-= 

super_dict = {}
for text in test:
    temp_dict = word_positions(text)
    for k, v in temp_dict.items():
        if super_dict.get(k) is None:
            super_dict[k] = []
        if v not in super_dict.get(k):
            super_dict[k].append(v)

# print(super_dict) # Check output