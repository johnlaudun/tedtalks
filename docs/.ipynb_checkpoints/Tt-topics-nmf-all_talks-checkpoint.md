# TEDtalks: `pandas` filters and visualizations

```python
>>> %pylab inline
Populating the interactive namespace from numpy and matplotlib
/Users/john/Library/Python/3.4/lib/python/site-packages/IPython/core/magics/pylab.py:161: UserWarning: pylab import has clobbered these variables: ['text']
`%matplotlib` prevents importing * from pylab and numpy
  "\n`%matplotlib` prevents importing * from pylab and numpy"
```

```python
>>> import pandas
>>> import re
...
>>> # Getting the data we need out of the CSV:
... colnames = ['author', 'title', 'date' , 'length', 'text']
>>> df = pandas.read_csv('../data/talks-v1b.csv', names=colnames)
>>> df['date'] = df['date'].replace(to_replace='[A-Za-z ]', value='', regex=True)
>>> talks = df.text.tolist()
>>> utalks = [unicode(i) for i in talks] # Clears unicode error in sk-learn vectorizer
>>> authors = df.author.tolist()
>>> dates = df.date.tolist()
>>> citations = [author+" "+date for author, date in zip(authors, dates)]
```

On stopwords in `sklearn`:

* According to the sklearn GH page on this topic the "list of English stop words is taken from the "Glasgow Information Retrieval Group". The original list can be found at http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words." [Link to GH page][].
* `sklearn` has a built-in list of stopwords for various languages. If, however, we decide not to use stopwords, and there's no reason why we couldn't take advantage of the inherent properties of TFIDF to eliminate common words, then the documentation notes: "If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0) to automatically detect and filter stop words based on intra corpus document frequency of terms."
* [David Blei][] did publish a list of stopwords with his _Turbo Topics_ python scripts. 
* A comparison of Blei's stopwords (296 words) and the Glasgow stopwords (318) revealed some slight differences -- Blei prefers to eliminate all uses of "use", for example, but that's about it.

[Link to GH page]: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/stop_words.py
[David Blei]: https://www.cs.princeton.edu/~blei/topicmodeling.html

```python
>>> import numpy as np
>>> import sklearn.feature_extraction.text as text
>>> from sklearn.decomposition import NMF
...
>>> # Getting topics using sklearn's NMF
...
... # Inputs
... n_samples = len(utalks)
>>> n_features = 1000
>>> n_topics = 35
>>> n_top_words = 10
...
>>> # Use tf-idf features for NMF.
... tfidf_vectorizer = text.TfidfVectorizer(max_df=0.95,
...                                         min_df=2,
...                                         max_features=n_features,
...                                         stop_words='english')
>>> tfidf = tfidf_vectorizer.fit_transform(utalks)
>>> tf_vectorizer = text.CountVectorizer(max_df=0.95,
...                                      min_df=2,
...                                      max_features=n_features,
...                                      stop_words='english')
>>> # Fit the NMF model
... print("Fitting the NMF model with {} topics for {} documents with {} features."
...       .format(n_topics, n_samples, n_features))
>>> nmf = NMF(n_components=n_topics,
...           random_state=1,
...           alpha=.1,
...           l1_ratio=.5).fit(tfidf)
Fitting the NMF model with 35 topics for 2113 documents with 1000 features.
```

```python
>>> print("\nTopics in NMF model:")
>>> features = tfidf_vectorizer.get_feature_names() # List of all features (words) in model
...
>>> # Function for printing topic words:
... def print_top_words(model, feature_names, n_top_words):
...     for topic_id, topic in enumerate(model.components_):
...         print('\nTopic {}:'.format(int(topic_id)))
...         print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
...               +', ' for i in topic.argsort()[:-n_top_words - 1:-1]]))
...
>>> print_top_words(nmf, features, n_top_words) #n_top_words can be changed on the fly
```

Create a dataframe:

    raw_data = {'student_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 
        'test_score': [76, 88, 84, 67, 53, 96, 64, 91, 77, 73, 52, np.NaN]}
    df = pd.DataFrame(raw_data, columns = ['student_name', 'test_score'])

To save a df to a CSV:

    DataFrame.to_csv(path_or_buf=None, sep=', ', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression=None, quoting=None, quotechar='"', line_terminator='\n', chunksize=None, tupleize_cols=False, date_format=None, doublequote=True, escapechar=None, decimal='.', **kwds)

```python
>>> # Now to associate NMF topics to documents...
...
... #tf = tf_vectorizer.fit_transform(utalks)
... dtm = tfidf.toarray()
>>> doctopic = nmf.fit_transform(dtm)
>>> print("Top NMF topics in...")
>>> for i in range(len(doctopic)):
...     top_topics = np.argsort(doctopic[i,:])[::-1][0:3]
...     top_topics_str = ' '.join(str(t) for t in top_topics)
...     print("{}: {}".format(citations[i], top_topics_str))
```

```python
>>> # Bar Chart of One Topic
... import matplotlib.pyplot as plt
...
>>> N, K = doctopic.shape
>>> ind = np.arange(N)
>>> width = 1
>>> plt.bar(ind, doctopic[:,0], width=width)
>>> plt.xticks(ind + width/2, citations) # put labels in the center
>>> plt.title('Share of Topic #0')
```

```python
>>> # Stacked Bar Chart of All Topics
... # Thanks to Alan Riddell (https://de.dariah.eu/tatom/topic_model_visualization.html)
...
... import numpy as np
>>> import matplotlib.pyplot as plt
...
>>> plt.figure(figsize=(12,8))
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

## Topics by Year

I have 35 topics, so 35 boxes: each with 10 bars.

```python
>>> doctopic = nmf.fit_transform(dtm)
>>> # Assuming that doctopic is a ARRAY!
... #
... # If this is actually an array, then you could just do:
... #      np.savetxt("foo.csv", doctopic, delimiter=",", fmt = "%s")
... # http://stackoverflow.com/questions/6081008/dump-a-numpy-array-into-a-csv-file
... #
... # The above won't give you the names of the files. Instead try this:
... # fileheader = np.concatenate((np.array([["citations"]]), np.array([list(range(K)])),axis = 1)
... #                                                         where K is the number of topics
... # docTopics = np.concatenate((citatations, doctopic), axis = 1)
... # docTopics = np.concatenate((fileheader, docTopics), axis = 0)
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
```
