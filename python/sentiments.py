#! /usr/bin/env python

'''
sentiments.py compares the outputs of the sentimental modules listed below.
Some normalization and smoothing has been attempted.
(I haven't implemented the NLTK solution because I don't have classified texts.)
'''

# Imports
import matplotlib.pyplot as plt
import seaborn # for more appealing plots
from nltk import tokenize
import numpy as np


# MPL Customizations
seaborn.set_style("darkgrid")
plt.rcParams['figure.figsize'] = 12, 8


# Below are three functions that break a text into sentences and then 
# assign a numerical value for sentiment based on a particular library.

## AFINN

def afinn_sentiment(filename):
    from afinn import Afinn
    afinn = Afinn()
    with open (my_file, "r") as myfile:
        text = myfile.read().replace('\n', ' ')
        sentences = tokenize.sent_tokenize(text)
        sentiments = []
        for sentence in sentences:
            sentsent = afinn.score(sentence)
            sentiments.append(sentsent)
        return sentiments


## TextBlob

def textblob_sentiment(filename):
    from textblob import TextBlob
    with open (filename, "r") as myfile:
        text=myfile.read().replace('\n', ' ')
        blob = TextBlob(text)
        textsentiments = []
        for sentence in blob.sentences:
            sentsent = sentence.sentiment.polarity
            textsentiments.append(sentsent)
        return textsentiments

## Indico

## Please note that my_key has been removed. Other users will need their own API key.

def indico_sentiment(filename):
    import indicoio
    indicoio.config.api_key = 'my_key'
    with open (my_file, "r") as myfile:
        text = myfile.read().replace('\n', ' ')
        sentences = tokenize.sent_tokenize(text)
        indico_sent = indicoio.sentiment(sentences)
    return indico_sent

# The next five functions are various mathematical calculations for
# averages and normalization (in order to make comparisons between 
# sentiment libraries.


## Moving Average with TA Library

def m_average(a_list, window):
    from talib import MA
    ma_array = np.asarray(a_list)
    return MA(ma_array,window)

## Running Mean with Numpy

def r_mean(a_list, window):
    rm_array = np.asarray(a_list)
    cumsum = np.cumsum(np.insert(rm_array, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window

## BONUS: Min-Max Function

def minmax(a_list):
    the_len  = len(a_list)
    min_val = min(a_list)
    max_val = max(a_list)
    the_range = max_val - min_val
    return (the_len, min_val, max_val, the_range)

## Normalization

def normed(a_list, norm_min, norm_max):
    old_min = min(a_list)
    old_max = max(a_list)
    old_range = old_max - old_min
    new_range = norm_max - norm_min
    output = [float((n - old_min) / old_range * new_range + norm_min) for n in a_list]
    return output

def mehrdad(a_list):
    mehrdad = a_list / np.linalg.norm(a_list)
    #mehrdad = a_list / np.max(np.abs(a_list))
    return mehrdad

# Plotting

# The five functions below are various plots: raw results, averaged, 
# normalized, normalized and averaged. 
# The function that calls the INDICO API is commented out because of
# the limits on the free version of the API.

def plot_sentiments(filename, annotation):
    fig = plt.figure()
    plt.title("Comparison of Sentiment Libraries")
    plt.plot(afinn_sentiment(filename), label = "Afinn")
    plt.plot(textblob_sentiment(filename), label = "TextBlob")
#    plt.plot(indico_sentiment(filename), label = "Indico")
    plt.ylabel("Emotional Valence")
    plt.xlabel("Sentence #")
    plt.legend(loc='lower right')
    plt.annotate(annotation, xy=(30, 2))

def avg_plots(filename, window):
    fig = plt.figure()
    plt.title("Averaged Sentiment")
    plt.plot(r_mean(afinn_sentiment(filename), window), label = "Afinn NP Running")
    plt.plot(r_mean(textblob_sentiment(filename), window), label = "TextBlob NP Running")
#    plt.plot(r_mean(indico_sentiment(filename), window), label = "Indico NP Running")
#    plt.plot(m_average(afinn_sentiment(filename), window), label = "Afinn TA Moving")
#    plt.plot(m_average(textblob_sentiment(filename), window), label = "TextBlob TA Moving")
#    plt.plot(m_average(indico_sentiment(filename), window), label = "Indico TA Moving")
    plt.ylabel("Emotional Valence")
    plt.xlabel("Sentence #")
    plt.legend(loc='lower center')

def normed_sentiment(filename):
    fig = plt.figure()
    plt.title("Comparison of Sentiment Libraries - Normalized")
    plt.plot(mehrdad(afinn_sentiment(filename)), label = "Afinn")
    plt.plot(mehrdad(textblob_sentiment(filename)), label = "TextBlob")
#    plt.plot(normed(indico_sentiment(filename), -1.0, 1.0), label = "Indico")
    plt.ylabel("Emotional Valence")
    plt.xlabel("Sentence #")
    plt.legend(loc='lower center')

def normavg_sentiment(filename, window):
    fig = plt.figure()
    plt.title("Comparison of Sentiment Libraries - Normalized and then Averaged Window={}".format(window))
    plt.plot(r_mean(normed(afinn_sentiment(filename), -1.0, 1.0), window), label = "Afinn")
    plt.plot(r_mean(normed(textblob_sentiment(filename), -1.0, 1.0), window), label = "TextBlob")
#    plt.plot(r_mean(normed(indico_sentiment(filename), -1.0, 1.0), window), label = "Indico")
    plt.ylabel("Emotional Valence")
    plt.xlabel("Sentence #")
    plt.legend(loc='lower center')

def mehrsent(filename, window):
    fig = plt.figure()
    plt.title("Comparison of Sentiment Libraries - Normalized and then Averaged Window={}".format(window))
    plt.plot(r_mean(mehrdad(afinn_sentiment(filename)), window), label = "Afinn")
    plt.plot(r_mean(mehrdad(textblob_sentiment(filename)), window), label = "TextBlob")
    plt.ylabel("Emotional Valence")
    plt.xlabel("Sentence #")
    plt.legend(loc='lower center')