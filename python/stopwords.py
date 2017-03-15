
# coding: utf-8

# In[11]:

#! /usr/bin/env python

import re
from stop_words import get_stop_words
from nltk.corpus import stopwords

# the two embedded lists
mod_stop = get_stop_words('en')
nltk_stop = stopwords.words("english")

# external lists
tt_list = re.split('\s+', open('../data/stopwords_tt.txt', 'r').read().lower())
mallet_list = re.split('\s+', open('../data/stopwords_mallet.txt', 'r').read().lower())
blei_list = re.split('\s+', open('../data/stopwords_Blei.txt', 'r').read().lower())

print("stop_words has {} words \n" 
      "the NLTK has {} words \n" 
      "the TedTalk list has {} words \n"
      "the MALLET list has {} words \n"
      "the Blei list has {} words".format(len(mod_stop), 
                                            len(nltk_stop), 
                                            len(tt_list), 
                                            len(mallet_list),
                                            len(blei_list)))

# Having determined that the best place to start is with the 
# combination of the Blei list and the stop_word list:

combo_list = mod_stop + blei_list
tt_stopset = set(combo_list)
tt_stoplist = sorted(list(tt_stop))
outfile = open('../data/tt_stop.txt', 'w')
outfile.write("\n".join(tt_stoplist))

