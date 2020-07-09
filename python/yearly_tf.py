#!/usr/bin/env python
"""
This script is a streamlined version of the 01-Trends-04-Yearly-Builds
notebook. It does the following:

* load the data
* filter the data to just the TED main talks
* group those talks by year to create term frequencies for each year year
* save the resulting dataframe to a csv

Some notes on how changes to the vectorizer parameters 
affect the overall word count:

min_df = 1, max_df = 1.0   ==> 39118
min_df = 2, max_df = n/a   ==> 21723
min_df = 2, max_df = 1.00  ==> 21723
min_df = 2, max_df = 0.99  ==> 19844 ("global" and "climate" disappear?!)
min_df = 2, max_df = 0.95  ==> 19844
min_df = 2, max_df = 0.90  ==> 19158

In the parameters section just below, min_years and max_years refer to the
minimum/maximum number of years in which a term must occur for it to be counted:
integers are actual number of years and floats are percentages. Default is 1 count minimum and 100 percent of years.
"""

# =-=-=-=-=-=-=-=-=-=-=
# USER OPTIONS
# =-=-=-=-=-=-=-=-=-=-= 

# Parameters for vectorizations
min_years = 2
max_years = 1.0

# File to load:
file_in  = '../output/TEDall.csv'
file_out = f'../output/yearly-tf-min{min_years}_max{max_years}.csv'

# =-=-=-=-=-=-=-=-=-=-=
# LOAD & COLLATE DATA
# =-=-=-=-=-=-=-=-=-=-= 

# Imports
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

# Load the Data
dfAll = pd.read_csv(file_in)

# Filter the dataframe to just the TED main talks:
main = dfAll[dfAll['Set']=='only']

# Concatenate all texts for a given year into one big pseudo-document:
all_years = main.groupby(['presented'])['text'].apply(lambda x: ','.join(x))

# Drop the first five years
years = all_years.drop([1984, 1990, 1994, 1998, 2001])

# =-=-=-=-=-=-=-=-=-=-=
# CLEAN THE TEXTS
# =-=-=-=-=-=-=-=-=-=-= 

parentheticals = [ "\(laughter\)", "\(applause\)", "\(music\)", 
                    "\(video\)", "\(laughs\)", "\(applause ends\)", 
                    "\(audio\)", "\(singing\)", "\(music ends\)", 
                    "\(cheers\)", "\(cheering\)", "\(recording\)", 
                    "\(beatboxing\)", "\(audience\)", "\(guitar strum\)", 
                    "\(clicks metronome\)", "\(sighs\)", "\(guitar\)", 
                    "\(marimba sounds\)", "\(drum sounds\)" ]



speakers = dfAll.speaker_1.tolist() + dfAll.speaker_2.tolist() + dfAll.speaker_3.tolist() + dfAll.speaker_4.tolist()


def remove_parens(text):
    new_text = text
    for rgx_match in parentheticals:
        new_text = re.sub(rgx_match, ' ', new_text.lower(), flags=re.IGNORECASE)
    return new_text

# Currently not working:
# `remove_speaker_names` keeps throwing a `TypeError`.

# def remove_speaker_names(text):
#     temp_text = text
#     for rgx_match in speakers:
#         temp_text = re.sub(rgx_match, ' ', temp_text)
#     return temp_text

# def clean_text(text):
#     the_text = text
#     cleaned = remove_parens(remove_speaker_names(the_text))
#     return cleaned


# =-=-=-=-=-=-=-=-=-=-=
# COUNT WORDS/TERMS
# =-=-=-=-=-=-=-=-=-=-= 

# Convert our series to a dataframe to make it easier to work in place:
dfYears = years.to_frame()

# Lowercase our texts
dfYears = dfYears.apply(lambda x: x.astype(str).str.lower())

# Remove everything that isn't a word, or space
dfYears = dfYears.replace('[^\w\s\+]', '', regex = True)

# Split on spaces and then count the length of the resulting list
dfYears['word_count'] = dfYears.text.apply(lambda x: len(str(x).split(' ')))


# =-=-=-=-=-=-=-=-=-=-=
# VECTORIZE
# =-=-=-=-=-=-=-=-=-=-= 

# Countvectorizer expects a list, so we create a list
texts = [ value for index, value in years.iteritems() ]

# We are going to bring our years back to the resulting term matrix below, 
# so while we are creating lists from our series, lets grab those years
# (And yes you can create two lists from one list comprehension, but don't.)
year_labels = [ index for index, value in years.iteritems() ]

# The usual incantation (minus the desired speaker removal for now):
vec = CountVectorizer(  preprocessor = remove_parens, 
                        min_df = min_years, 
                        max_df = max_years
                        )
word_count_vectors = vec.fit_transform(texts)

# Create a dataframe from the resulting array
X = vec.fit_transform(texts)
term_matrix = pd.DataFrame(X.todense(), columns=vec.get_feature_names())

# Label the rows of our dataframe with labels from above
term_matrix['year'] = year_labels

# Set the index to our newly created column 
term_matrix.set_index('year', inplace = True)

# Transpose our dataframe so that years are columns and words are rows
term_df = term_matrix.transpose()
term_df.reset_index(inplace=True)
word_df = term_df.rename(columns={'index': 'term'})

# Save the dataframe 
word_df.to_csv(file_out)

print(f'CSV created with {term_df.shape} shape.')