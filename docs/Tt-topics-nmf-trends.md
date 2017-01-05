# TEDtalk Trends

This is my attempt to work my way back into the main lines of our research project. I am working, once again, from the [Dariah][] materials prepared, I think, by Alan Riddell.

[Dariah]: https://de.dariah.eu/tatom/visualizing_trends.html

Since I can't remember if we saved our previous versions of topics, and since this is an experiment for the time being in visualizing trends, I am going to work through the `pandas` filtering process and then the NMF topic modeling -- though I should note there is now an [LDA][] module for Python. 

[LDA]: https://ariddell.org/lda.html

The Dariah post is focused on tracking a single topic through a text: in this case, the "revolutionary" topic through Victor Hugo's 500,000 word _Les Miserables_. The words there are all in a fixed sequence. We don't quite have that, rather, we have 10 groups of words, for each of ten years. Any sequencing within a given year would, I think be entirely arbitrary, so we are going to have a graph with only 10 points on the x-axis? Maybe better visualized as 10 bars?

```python
>>> %pylab inline
Populating the interactive namespace from numpy and matplotlib
```

```python
>>> import pandas
>>> import numpy as np
>>> import sklearn.feature_extraction.text as text
>>> from sklearn.decomposition import NMF
...
>>> # =-=-=-=-=-=
... # Get the data we need out of the CSV
... # =-=-=-=-=-=
...
... colnames = ['author', 'title', 'date' , 'length', 'text']
>>> df = pandas.read_csv('../data/talks-v1b.csv', names=colnames)
>>> df['date'] = df['date'].replace(to_replace='[A-Za-z ]', value='', regex=True)
>>> talks = df.text.tolist()
>>> utalks = [unicode(i) for i in talks] # Clears unicode error in sk-learn vectorizer
>>> authors = df.author.tolist()
>>> dates = df.date.tolist()
>>> citations = [author+" "+date for author, date in zip(authors, dates)]
...
>>> # =-=-=-=-=-=
... # Get topics using sklearn's NMF
... # =-=-=-=-=-=
...
... # All our variables are here to make it easier to make adjustments
... n_samples = len(utalks)
>>> n_features = 1000
>>> n_topics = 45
>>> n_top_words = 15
>>> tt_stopwords = open('../data/tt_stopwords.txt', 'r').read().splitlines()
...
>>> # Get tf-idf features for NMF
... tfidf_vectorizer = text.TfidfVectorizer(max_df = 0.95,
...                                         min_df = 2,
...                                         max_features = n_features,
...                                         stop_words = tt_stopwords)
>>> tfidf = tfidf_vectorizer.fit_transform(utalks)
...
>>> # Fit the NMF model
... nmf = NMF(n_components=n_topics,
...           random_state=1,
...           alpha=.1,
...           l1_ratio=.5).fit(tfidf)
>>> print("Fitting the NMF model with {} topics for {} documents with {} features."
...       .format(n_topics, n_samples, n_features))
Fitting the NMF model with 45 topics for 2113 documents with 1000 features.
```

```python
>>> dtm = tfidf.toarray()
>>> doctopic = nmf.fit_transform(dtm) # This is an array
...
>>> # =-=-=-=-=-=
... # Creating arrays in the order that we want them
... # =-=-=-=-=-=
...
... # The AUTHORS, DATES, and CITATIONS variables are lists. We need them to be arrays.
... authors = np.array([authors])  # Note: the seemingly extra [] make the array have
>>> dates = np.array([dates])      #       dimensions of (k,1) instead of just (k,)
>>> citations = np.array([citations])
```

```python
>>> type(dates)
numpy.ndarray
```

```python
>>> dates.shape
(1, 2113)
```

```python
>>> # Below syntax is from: http://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
...
... # We want to sort everything by the dates. To do this, we will use ARGSORT() to
... # give us the indices by which to sort all of our other arrays:
...
... ##JL: This is like magic to my mind:
...
... inds = np.argsort(dates)
...
>>> print(dates.shape)
>>> print(authors.shape)
>>> print(doctopic.shape)
>>> print(inds.shape)
(1, 2113)
(1, 2113)
(2113, 45)
(1, 2113)
```

```python
>>> # Now we will use INDS to sort each of our arrays: DATES, AUTHORS, and DOCTOPIC:
... dates = dates[:,inds]
>>> authors = authors[:,inds]
>>> doctopic = doctopic[inds,:]
```

```python
>>> # =-=-=-=-=-=
... # Saving output to CSV
... # =-=-=-=-=-=
...
... # Since DOCTOPIC is an array, you can just do:
... #      np.savetxt("foo.csv", doctopic, delimiter=",", fmt = "%s")
... # http://stackoverflow.com/questions/6081008/dump-a-numpy-array-into-a-csv-file
... #
... # The above won't give you the names of the files. Instead try this:
...
... topsnum = np.array([list(range(n_topics))])
>>> # topsnum = np.indices((1,n_topics))[1] <-- this is more than we need,
... #                                           but it's cool to know more tricks
... #
... # Two ways to get an array that is of the form [[0,1,2,3,...]].
... # It will have the desired dimensions of (1,35) which is what we want
...
...
... fileheader = np.concatenate((np.array([["citations"]]), topsnum),axis = 1)
...
>>> docTopics = np.concatenate((citations, doctopic), axis = 1)
>>> docTopics = np.concatenate((fileheader, docTopics), axis = 0)
...
>>> np.savetxt("../data/sortedtalks.csv", docTopics, delimiter=",", fmt = "%s")
...
>>> # =-=-=-=-=-=
... # Finding where to cut the data to get one dataset per year
... # =-=-=-=-=-=
...
... # Here we create a smaller fileheader that will only include the authors.
...
... yfileheader = np.concatenate((np.array([["authors"]]), topsnum),axis = 1)
...
>>> minyear = dates[0]
>>> maxyear = dates[-1]
...
>>> # We initialize STARTINDS at 0 because our dates are already sorted.
... startinds = 0
...
>>> for year in range(minyear, maxyear):
...     lastinds = np.searchsorted(dates, year + 1, "left")
...     # The SEARCHSORTED command looks for values in a sorted list.
...     # The "LEFT" flag just gives us the index for the first instance.
...
...     # Slice the rows that we need for this YEAR.
...     yearDT = doctopic[startinds:lastinds,:]
...     yearA = authors[startinds:lastinds]
...
...     # Add the pieces back together
...     yearDTwA = np.concatenate((yearA,yearDT), axis = 1)
...     yearDTwApH = np.concatenate((yfileheader, yearDTwA), axis = 0)
...
...     # Create FILENAME and save the data
...     filename = "talks_year" + str(year) + ".csv"
...     np.savetxt(filename, yearDTwApH, delimiter=",", fmt = "%s")
```

```python
>>> # =-=-=-=-=-=
... # Understanding why and how the printing of the topics works.
... # =-=-=-=-=-=
...
... for i in range(len(doctopic)): #march over each row --> document
...     top_topics = np.argsort(doctopic[i,:])[::-1][0:3]
...     # DOCTOPIC[I,:] is the amount of each topic starting at 0
...     # NP.ARGSORT(DOCTOPIC[I,:]) tells us how topics are used starting with the least
...     #        EX - test = np.matrix("1 3 4 2")
...     #        	  np.argsort(test) will yield matrix([[0, 3, 1, 2]])
...     # [::-1] reverses the order.
...     # So NP.ARGSORT(DOCTOPIC[I,:])[::-1] gives us the topics in order of being used
...     #        biggest to smallest
...     # Then NP.ARGSORT(DOCTOPIC[I,:])[::-1][0:3] gives us the top 3 topics
...     top_topics_str = ' '.join(str(t) for t in top_topics)
...     # Here you loop over each of the top three topics and convert the integer type to a
...     # string and concatenates all of these together.
...     print("{}: {}".format(citations[i], top_topics_str))
...     # Then you print two strings for each document
...
>>> """
>>> Ok so we don't need this, but I already wrote it before I figured that out.
>>> So I'm leaving it in...
...
>>> # Pre-allocate the YEARS variable
... years = np.array([np.zeros(n_samples)]).T
...
>>> for i in range(len(doctopic)): #march over each row --> document
...     years[i] = citations[i].split()[-1]
...     # CITATIONS is a list of strings
...     # .SPLIT() splits the strings at the space.
...     # [-1] takes the last element in the list
>>> """
```

```python
>>> # Stacked Bar Chart of All Topics
... # Thanks to Alan Riddell (https://de.dariah.eu/tatom/topic_model_visualization.html)
...
... import numpy as np
>>> import matplotlib.pyplot as plt
...
>>> plt.figure(figsize=(18,8))
>>> #fig = matplotlib.pyplot.gcf()
... #fig.set_size_inches(18.5, 10.5)
...
... N, K = doctopic.shape  # N documents, K topics
>>> ind = np.arange(N)  # the x-axis locations for the texts
>>> width = 1  # the width of the bars
>>> plots = []
>>> height_cumulative = np.zeros(N)
...
>>> for k in range(K):
...     color = plt.cm.coolwarm(k/K, 1)
...     if k == 0:
...         p = plt.bar(ind, doctopic[:, k], width, color=color)
...     else:
...         p = plt.bar(ind, doctopic[:, k], width, bottom=height_cumulative, color=color)
...     height_cumulative += doctopic[:, k]
...     plots.append(p)
...
>>> plt.ylim((0, 1))  # proportions sum to 1, so the height of the stacked bars is 1
>>> plt.ylabel('Topics')
>>> plt.title('Topics in 2006 TEDtalks')
>>> plt.xticks(ind+width/2, citations, rotation='vertical')
>>> plt.yticks(np.arange(0, 1, 10))
>>> topic_labels = ['Topic #{}'.format(k) for k in range(K)]
>>> plt.legend([p[0] for p in plots], topic_labels)
>>> plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
>>> plt.show()
```
