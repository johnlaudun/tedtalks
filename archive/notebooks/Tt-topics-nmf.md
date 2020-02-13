# TEDtalks: `pandas` filters and visualizations

```python
 %pylab inline
Populating the interactive namespace from numpy and matplotlib
```

## Pandas filters


 import pandas
 import re
 colnames = ['author', 'title', 'date' , 'length', 'text']
 df = pandas.read_csv('../data/talks-v1b.csv', names=colnames)
...
 df['date'] = df['date'].replace(to_replace='[A-Za-z ]', value='', regex=True)
...
 data2016 = df[df['date'] == '2016']
...
 talks = data2016.text.tolist()
 authors = data2016.author.tolist()
 dates = data2016.date.tolist()
...
 # Combining year with presenter for citations
... citations = [author+" "+date for author, date in zip(authors, dates)]


 import numpy as np
 import sklearn.feature_extraction.text as text
 from sklearn.decomposition import NMF, LatentDirichletAllocation
...
 # Function for printing topic words (used later):
...
 n_samples = len(talks)
 n_features = 1000
 n_topics = 45
 n_top_words = 15
...
 # Use tf-idf features for NMF.
... tfidf_vectorizer = text.TfidfVectorizer(max_df=0.95, min_df=2,
...                                    max_features=n_features,
...                                    stop_words='english')
 tfidf = tfidf_vectorizer.fit_transform(talks)
 tf_vectorizer = text.CountVectorizer(max_df=0.95, min_df=2,
...                                 max_features=n_features,
...                                 stop_words='english')
...
 # Use tf (raw term count) features for LDA.
... # tf = tf_vectorizer.fit_transform(talks)

 # Fit the NMF model
... print("Fitting the NMF model with tf-idf features, "
...       "n_topics={}, n_samples={} and n_features={}...".format(n_topics, n_samples, n_features))
 nmf = NMF(n_components=n_topics,
...           random_state=1,
...           alpha=.1,
...           l1_ratio=.5).fit(tfidf)
Fitting the NMF model with tf-idf features, n_topics=45, n_samples=95 and n_features=1000...


 # Now to associate NMF topics to documents...
... tf = tf_vectorizer.fit_transform(talks)
 dtm = tf.toarray()
 doctopic = nmf.fit_transform(dtm)
 print("Top NMF topics in...")
 for i in range(len(doctopic)):
...     top_topics = np.argsort(doctopic[i,:])[::-1][0:3]
...     top_topics_str = ' '.join(str(t) for t in top_topics)
...     print("{}: {}".format(citations[i], top_topics_str))
Top NMF topics in...
Harry Cliff 2016: 20 2 30
Sebastian Wernicke 2016: 5 4 38
Aomawa Shields 2016: 39 40 1
David Sedlak 2016: 1 23 26
James Veitch 2016: 41 11 19
Tim Harford 2016: 0 41 38
Melvin Russell 2016: 19 2 3
Wael Ghonim 2016: 41 22 29
Ole Scheeren 2016: 22 1 5
Jill Farrant 2016: 14 1 24
Oscar Schwartz 2016: 12 3 29
Achenyo Idachaba 2016: 1 19 14
Elizabeth Lev 2016: 40 22 34
Yanis Varoufakis 2016: 33 1 39
David Gruber 2016: 40 18 10
Tania Simoncelli 2016: 9 5 41
Auke Ijspeert 2016: 17 28 2
Melati and Isabel Wijsen 2016: 43 41 16
Linda Liukas 2016: 30 40 3
Andrés Ruzo 2016: 35 1 30
Judson Brewer 2016: 4 17 0
Pardis Sabeti 2016: 43 44 5
Matthew Williams 2016: 28 44 11
Dambisa Moyo 2016: 8 39 3
Sean Follmer 2016: 0 22 42
Gregory Heyworth 2016: 40 27 2
Mike Velings 2016: 10 34 14
Dorothy Roberts 2016: 15 12 30
Jocelyne Bloch 2016: 4 31 17
Celeste Headlee 2016: 41 12 17
Shonda Rhimes 2016: 38 2 28
Allan Adams 2016: 20 3 39
Raffaello D'Andrea 2016: 30 20 22
Al Gore 2016: 34 2 3
Dalia Mogahed 2016: 41 3 29
Audrey Choi 2016: 0 2 13
Mary Bassett 2016: 15 25 34
Ivan Coyote 2016: 41 36 42
Thomas Peschak 2016: 18 10 35
Magda Sayeg 2016: 23 22 0
Russ Altman 2016: 11 3 2
Alexander Betts 2016: 7 37 44
Travis Kalanick 2016: 23 2 12
Reshma Saujani 2016: 16 34 0
Caleb Harper 2016: 14 41 30
Laura Robinson 2016: 18 13 44
Mileha Soneji 2016: 17 30 41
Tshering Tobgay 2016: 25 39 2
Joe Gebbia 2016: 42 41 3
Tim Urban 2016: 26 4 44
Jessica Ladd 2016: 0 29 30
Arthur Brooks 2016: 41 2 8
Meron Gribetz 2016: 30 4 42
Adam Foss 2016: 36 12 8
Carol Fishman Cohen 2016: 44 21 19
Latif Nasser 2016: 35 34 40
Siyanda Mohutsiwa 2016: 27 5 8
Alex Kipman 2016: 30 6 44
Angélica Dass 2016: 40 15 44
Dan Gross 2016: 29 21 5
Lisa Nip 2016: 6 35 39
Knut Haanaes 2016: 0 44 23
Adam Grant 2016: 41 40 23
Haley Van Dyck 2016: 43 12 21
Parag Khanna 2016: 2 1 12
Danielle Feinberg 2016: 40 1 38
Tabetha Boyajian 2016: 39 3 5
Robert Palmer 2016: 0 3 37
Linus Torvalds 2016: 3 39 44
Hugh Evans 2016: 13 1 12
Stephen Petranek 2016: 6 41 3
Paula Hammond 2016: 31 24 11
Astro Teller 2016: 23 42 40
Mary Norris 2016: 34 0 20
Christiana Figueres 2016: 37 34 25
Joshua Prager 2016: 32 35 36
Chris Anderson 2016: 41 4 22
Juan Enriquez 2016: 26 31 2
Aditi Gupta 2016: 16 27 44
Kenneth Lacovara 2016: 35 3 41
Shivani Siroya 2016: 5 41 36
R. Luke DuBois 2016: 26 40 23
Ameera Harouda 2016: 25 22 36
Michael Metcalfe 2016: 37 3 36
Riccardo Sabatini 2016: 32 3 2
Sarah Gray 2016: 11 40 21
Alice Rawsthorn 2016: 42 27 22
Dan Pallotta 2016: 30 41 38
Monica Byrne 2016: 35 41 20
Jennifer Kahn 2016: 24 41 3
Uri Hasson 2016: 4 21 12
Sanford Biggers 2016: 40 15 44
Sangeeta Bhatia 2016: 31 40 21
Kang Lee 2016: 21 3 44
Moran Cerf 2016: 4 40 3
```

```python
 doctopic.shape
```

```python
 doctopic
```

```python
 # Bar Chart of One Topic
... import matplotlib.pyplot as plt
...
 N, K = doctopic.shape
 ind = np.arange(N)
 width = 1
 plt.bar(ind, doctopic[:,0], width=width)
 plt.xticks(ind + width/2, citations) # put labels in the center
 plt.title('Share of Topic #0')
```

```python
 # Stacked Bar Chart of All Topics
... # Thanks to Alan Riddell (https://de.dariah.eu/tatom/topic_model_visualization.html)
...
... import numpy as np
 import matplotlib.pyplot as plt
...
 plt.figure(figsize=(24,8))
 #fig = matplotlib.pyplot.gcf()
... #fig.set_size_inches(18.5, 10.5)
...
... N, K = doctopic.shape  # N documents, K topics
 ind = np.arange(N)  # the x-axis locations for the texts
 width = 1  # the width of the bars
 plots = []
 height_cumulative = np.zeros(N)
...
 for k in range(K):
...     color = plt.cm.coolwarm(k/K, 1)
...     if k == 0:
...         p = plt.bar(ind, doctopic[:, k], width, color=color)
...     else:
...         p = plt.bar(ind, doctopic[:, k], width, bottom=height_cumulative, color=color)
...     height_cumulative += doctopic[:, k]
...     plots.append(p)
...
 plt.ylim((0, 1))  # proportions sum to 1, so the height of the stacked bars is 1
 plt.ylabel('Topics')
 plt.title('Topics in 2016 TEDtalks')
 plt.xticks(ind+width/2, citations, rotation='vertical')
 plt.yticks(np.arange(0, 1, 10))
 topic_labels = ['Topic #{}'.format(k) for k in range(K)]
 #plt.legend([p[0] for p in plots], topic_labels)
... #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
... plt.show()
```

## Working with LSI/LSA to get Elbow graph for k-means

I got the idea from [Sujit Pal][].

```python
 from sklearn.decomposition import TruncatedSVD
 from sklearn.preprocessing import Normalizer
...
 lsa = TruncatedSVD()
 my_lsa = lsa.fit_transform(tfidf)
 #normed = Normalizer(copy=False).fit_transform(my_lsa)
... #cosine_similarity(normed)
```

```python
 print(normed)
```

```python
 # write out coordinates to file
... fcoords = open(os.path.join(MODELS_DIR, "coords.csv"), 'wb')
 for vector in lsi[corpus]:
...     if len(vector) != 2:
...         continue
...     fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
 fcoords.close()
```
