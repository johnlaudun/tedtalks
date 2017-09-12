# TEDtalks: `pandas` filters and visualizations

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
```

```python
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
>>> # Below sytax is from: http://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
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
>>> np.savetxt("sortedtalks.csv", docTopics, delimiter=",", fmt = "%s")
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
['Al Gore 2006' 'David Pogue 2006' 'Cameron Sinclair 2006' ...,
 'Justin Hall-Tipping 2011' 'Guy-Philippe Goldstein 2011'
 'Todd Kuiken 2011']: [22 21 24 25 26 27 28 19 29 34 36 37 39 41 42 30 18 44 16 15  1  2  3 13  4
 11 10  9  8  7 40 31  5  6 43 14 23 20 38 35 17  0 12 33 32] [22 21 24 25 26 27 28 30 31 33 34 36 37 38 39 42 20 18 44  8  7  4 17  9  3
 10  6 11 13 14  1 15 12  5 40 16 19 29 23 32 35 43  0  2 41] [22 24 25 27 28 29 30 31 32 23 33 35 36 37 38 39 40 41 42 34 43 44 12  4  5
  6  7  8  9 21 11  1 13 14 15 16 17 18 19  3 26 20 10  2  0]
"\nOk so we don't need this, but I already wrote it before I figured that out.\nSo I'm leaving it in...\n\n# Pre-allocate the YEARS variable\nyears = np.array([np.zeros(n_samples)]).T\n\nfor i in range(len(doctopic)): #march over each row --> document\n    years[i] = citations[i].split()[-1]\n    # CITATIONS is a list of strings\n    # .SPLIT() splits the strings at the space.\n    # [-1] takes the last element in the list\n"
```

```python
>>> # =-=-=-=-=-=
... # Create a network with the topics as the nodes and edges weighted by the
... # number of documents that have these topics as the K-top topics
... # =-=-=-=-=-=
...
... # Step 1 - Create a matrix of zeros and ones where each row is a document and
... #          each column is a topic. An entry of 1 denotes that that topic is
... #          one of the top K ones in that document.
...
... # K_TOPICS is just the number of top topics that you consider per document
... k_topics = 3
...
>>> # Preallocate DOC_BY_TOPICS matrix to have N_SAMPLES rows and N_TOPICS columns
... doc_by_topics = np.array(np.zeros((n_samples,n_topics), dtype = np.int))
...
>>> for i in range(len(doctopic)):
...     # Pull the top K topics, just like above
...     top_topics = np.argsort(doctopic[i,:])[::-1][0:k_topics]
...     # Set those topics to 1 in the documents row of zeros
...     doc_by_topics[i,top_topics] = 1 # Topics are on or off
...
...     # DOC_BY_TOPICS: rows are documents, columns are topics
...
>>> # Step 2 - Pairwise compare topics
...
... # Preallocate SSM_TOPICS and JAC_TOPICS matrices to be square matrices with
... #            N_TOPICS rows and columns.
... # SSM will just be the number of documents where they overlap
... # JAC will be the jaccard similarity
... ssm_topics = np.array(np.zeros((n_topics,n_topics), dtype = np.int))
>>> jac_topics = np.array(np.zeros((n_topics,n_topics)))
...
>>> for i in range(n_topics):
...     for j in range(i+1, n_topics):
...         # First, we overlap the columns associated to the I^th topic and to
...         #        the J^th one.
...         sum_doc = doc_by_topics[:,i]+doc_by_topics[:,j]
...
...         # Using SUM_DOC, we will determine what documents share these two
...         # topics as top documents
...         # CAP_DOC comes from the name for the symbol denoting intersection
...         cap_doc = (sum_doc == 2).sum()
...         # Line based off: http://stackoverflow.com/questions/10525921/numpy-array-boolean-to-binary
...
...         # Now we will determine when either topic occurs in the top K of
...         # topics for each document
...         # CUP_DOC comes from the name for the symbol denoting union
...         cup_doc = cap_doc + (sum_doc == 1).sum()
...
...         ssm_topics[i,j] = cap_doc
...         jac_topics[i,j] = cap_doc/cup_doc
...
>>> # Since our measure of similiarity is symmetric, we can add the transpose of
... # the upper-triangular matrices that were created above to the original matrix
... # to get the whole SSM or JAC matrix.
... ssm_topics = ssm_topics + ssm_topics.T
>>> jac_topics = jac_topics + jac_topics.T
```

```python
>>> import networkx as nx
>>> import matplotlib.pyplot as plt
...
>>> # Create a network from the matrix
... G = nx.from_numpy_matrix(jac_topics)
...
>>> # Draw the graph
... pos = nx.spring_layout(G, k=0.20, iterations=1)
>>> fig = plt.figure(figsize=(15,15))
>>> nx.draw_networkx_nodes(G,pos,node_size=20,node_color='w',alpha=0.4)
>>> nx.draw_networkx_labels(G,pos,fontsize=14)
>>> font = {'fontname'   : 'Helvetica',
...             'color'      : 'k',
...             'fontweight' : 'bold',
...             'fontsize'   : 14}
>>> nx.draw(G, pos)
>>> plt.show()
```

```python
>>> # Code from: https://networkx.github.io/documentation/networkx-1.10/examples/drawing/degree_histogram.html
... degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
>>> #print "Degree sequence", degree_sequence
... dmax=max(degree_sequence)
...
>>> plt.loglog(degree_sequence,'b-',marker='o')
>>> plt.title("Degree rank plot")
>>> plt.ylabel("degree")
>>> plt.xlabel("rank")
...
>>> # draw graph in inset
... plt.axes([0.45,0.45,0.45,0.45])
>>> Gcc=sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]
>>> pos=nx.spring_layout(Gcc)
>>> plt.axis('off')
>>> nx.draw_networkx_nodes(Gcc,pos,node_size=20)
>>> nx.draw_networkx_edges(Gcc,pos,alpha=0.4)
...
>>> plt.show()
```
