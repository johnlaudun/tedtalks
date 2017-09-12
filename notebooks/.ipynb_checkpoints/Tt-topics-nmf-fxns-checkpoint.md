# TEDtalks: `pandas` filters and visualizations

```python
>>> %pylab inline
Populating the interactive namespace from numpy and matplotlib
```

## Pandas filters

```python
>>> def talks_for_year(year):
...     import pandas
...     import re
...     colnames = ['author', 'title', 'date' , 'length', 'text']
...     df = pandas.read_csv('../data/talks-v1b.csv', names=colnames)
...     df['date'] = df['date'].replace(to_replace='[A-Za-z ]', value='', regex=True)
...     datayear = df[df['date'] == '{:d}'.format(year)]
...     talks = datayear.text.tolist()
...     utalks = [unicode(i) for i in talks]
...     return utalks
...
>>> def citations_for_year(year):
...     import pandas
...     import re
...     colnames = ['author', 'title', 'date' , 'length', 'text']
...     df = pandas.read_csv('../data/talks-v1b.csv', names=colnames)
...     df['date'] = df['date'].replace(to_replace='[A-Za-z ]', value='', regex=True)
...     datayear = df[df['date'] == '{:d}'.format(year)]
...     authors = datayear.author.tolist()
...     dates = datayear.date.tolist()
...     # Combining year with presenter for citations
...     citations = [author+" "+date for author, date in zip(authors, dates)]
...     return citations
```

```python
>>> talks2007 = talks_for_year(2007)
>>> citations2007 = citations_for_year(2007)
```

## NMF Topics

```python
>>> def nmf_model(corpus, topics, features):
...     """Topic Model using NMF: requires corpus, topics, and features."""
...     import numpy as np
...     import sklearn.feature_extraction.text as text
...     from sklearn.decomposition import NMF
...     n_samples = len(corpus)
...     n_features = features
...     n_topics = topics
...     # Use tf-idf features for NMF.
...     tfidf_vectorizer = text.TfidfVectorizer(max_df=0.95,
...                                             min_df=2,
...                                             max_features=n_features,
...                                             stop_words='english')
...     tfidf = tfidf_vectorizer.fit_transform(corpus)
...     tf_vectorizer = text.CountVectorizer(max_df=0.95,
...                                          min_df=2,
...                                          max_features=n_features,
...                                          stop_words='english')
...     # Fit the NMF model
...     print("Fitting the NMF model with {} topics for {} documents with {} features.".format(n_topics, n_samples, n_features))
...     nmf = NMF(n_components=n_topics,
...               random_state=1,
...               alpha=.1,
...               l1_ratio=.5).fit(tfidf)
...     return
```

```python
>>> nmf_model(talks2007, 15, 1000)
Fitting the NMF model with 15 topics for 119 documents with 1000 features.
```

```python
>>> print("\nTopics in NMF model:")
>>> tfidf_feature_names = tfidf_vectorizer.get_feature_names()
...
>>> # Function for printing topic words (used later):
... def print_top_words(model, feature_names, n_top_words):
...     for topic_id, topic in enumerate(model.components_):
...         print('\nTopic {}:'.format(int(topic_id)))
...         print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
...               +', ' for i in topic.argsort()[:-n_top_words - 1:-1]]))
...
>>> print_top_words(nmf, tfidf_feature_names, n_top_words) #n_top_words can be changed on the fly
```

```python
>>> # Now to associate NMF topics to documents...
... dtm = tf.toarray()
>>> doctopic = nmf.fit_transform(dtm)
>>> print("Top NMF topics in...")
>>> for i in range(len(doctopic)):
...     top_topics = np.argsort(doctopic[i,:])[::-1][0:3]
...     top_topics_str = ' '.join(str(t) for t in top_topics)
...     print("{}: {}".format(citations[i], top_topics_str))
```

```python
>>> doctopic.shape
```

```python
>>> doctopic
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
