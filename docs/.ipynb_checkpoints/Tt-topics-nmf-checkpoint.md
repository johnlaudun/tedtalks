# TEDtalks: `pandas` filters and visualizations

```python
>>> %pylab inline
Populating the interactive namespace from numpy and matplotlib
```

## Pandas filters

```python
>>> import pandas
>>> import re
>>> colnames = ['author', 'title', 'date' , 'length', 'text']
>>> df = pandas.read_csv('../data/talks-v1b.csv', names=colnames)
...
>>> df['date'] = df['date'].replace(to_replace='[A-Za-z ]', value='', regex=True)
...
>>> data2006 = df[df['date'] == '2006']
...
>>> talks = data2006.text.tolist()
>>> authors = data2006.author.tolist()
>>> dates = data2006.date.tolist()
...
>>> # Combining year with presenter for citations
... citations = [author+" "+date for author, date in zip(authors, dates)]
```

```python
>>> import numpy as np
>>> import sklearn.feature_extraction.text as text
>>> from sklearn.decomposition import NMF, LatentDirichletAllocation
...
>>> # Function for printing topic words (used later):
... def print_top_words(model, feature_names, n_top_words):
...     for topic_id, topic in enumerate(model.components_):
...         print('\nTopic {}:'.format(int(topic_id)))
...         print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
...               +', ' for i in topic.argsort()[:-n_top_words - 1:-1]]))
...
>>> n_samples = len(talks)
>>> n_features = 1000
>>> n_topics = 15
>>> n_top_words = 15
...
>>> # Use tf-idf features for NMF.
... tfidf_vectorizer = text.TfidfVectorizer(max_df=0.95, min_df=2,
...                                    max_features=n_features,
...                                    stop_words='english')
>>> tfidf = tfidf_vectorizer.fit_transform(talks)
>>> tf_vectorizer = text.CountVectorizer(max_df=0.95, min_df=2,
...                                 max_features=n_features,
...                                 stop_words='english')
...
>>> # Use tf (raw term count) features for LDA.
... tf = tf_vectorizer.fit_transform(talks)
```

```python
>>> # Fit the NMF model
... print("Fitting the NMF model with tf-idf features, "
...       "n_topics={}, n_samples={} and n_features={}...".format(n_topics, n_samples, n_features))
>>> nmf = NMF(n_components=n_topics,
...           random_state=1,
...           alpha=.1,
...           l1_ratio=.5).fit(tfidf)
Fitting the NMF model with tf-idf features, n_topics=15, n_samples=50 and n_features=1000...
```

```python
>>> print("\nTopics in NMF model:")
>>> tfidf_feature_names = tfidf_vectorizer.get_feature_names()
>>> print_top_words(nmf, tfidf_feature_names, n_top_words) #n_top_words can be changed on the fly

Topics in NMF model:

Topic 0:
people 0.33, know 0.23, world 0.2, ve 0.17, said 0.16, things 0.16, say 0.15, years 0.15, want 0.14, time 0.13, got 0.12, actually 0.12, ll 0.1, lot 0.1, work 0.09, 

Topic 1:
countries 0.26, data 0.26, africa 0.2, world 0.2, families 0.12, country 0.12, income 0.11, aid 0.1, asia 0.1, health 0.1, uganda 0.09, students 0.08, south 0.07, america 0.07, economy 0.07, 

Topic 2:
interface 0.31, multi 0.22, kind 0.19, touch 0.17, hands 0.17, data 0.1, things 0.09, use 0.08, nasa 0.07, lab 0.07, tool 0.06, research 0.05, control 0.04, expect 0.04, points 0.04, 

Topic 3:
technology 0.53, evolution 0.22, life 0.16, game 0.15, actually 0.12, machines 0.09, biological 0.08, technologies 0.08, species 0.07, things 0.06, general 0.05, wants 0.04, took 0.04, price 0.03, term 0.03, 

Topic 4:
cancer 0.53, muscle 0.45, cells 0.26, tumor 0.16, body 0.11, blood 0.07, cell 0.04, articles 0.04, brain 0.02, reading 0.02, disease 0.01, growth 0.01, medical 0.01, tried 0.01, percent 0.0, 

Topic 5:
draw 0.27, choose 0.2, drawing 0.16, theme 0.14, piece 0.13, main 0.1, decide 0.09, play 0.08, romantic 0.08, ok 0.07, try 0.07, supposed 0.07, happens 0.07, ll 0.06, know 0.06, 

Topic 6:
film 0.36, people 0.15, arab 0.14, independent 0.12, world 0.11, wish 0.11, al 0.07, peace 0.07, feeling 0.07, war 0.07, incredible 0.06, photographs 0.05, military 0.05, power 0.05, know 0.04, 

Topic 7:
bronx 0.36, south 0.19, community 0.16, city 0.16, environmental 0.15, sustainable 0.09, waste 0.09, development 0.09, green 0.08, communities 0.08, justice 0.08, common 0.07, urban 0.06, dog 0.05, black 0.05, 

Topic 8:
com 0.35, internet 0.3, email 0.17, boy 0.1, dead 0.09, listen 0.08, web 0.08, known 0.08, guess 0.07, attachment 0.06, pole 0.06, possible 0.06, interface 0.06, shit 0.05, maybe 0.04, 

Topic 9:
love 0.35, romantic 0.25, women 0.24, brain 0.22, sex 0.16, attachment 0.14, men 0.09, moving 0.08, drive 0.08, fall 0.07, somebody 0.05, kill 0.05, person 0.05, market 0.04, societies 0.04, 

Topic 10:
heart 0.28, device 0.26, attack 0.25, signal 0.21, patient 0.19, electrical 0.16, brain 0.15, patients 0.1, magnetic 0.08, emergency 0.06, save 0.06, st 0.05, wish 0.04, medical 0.03, means 0.03, 

Topic 11:
gang 0.75, cocaine 0.18, crack 0.14, mcdonald 0.1, inner 0.05, death 0.04, selling 0.03, city 0.02, soldiers 0.02, shit 0.01, drugs 0.01, money 0.01, job 0.01, willing 0.01, got 0.01, 

Topic 12:
sauce 0.61, tomato 0.26, pepsi 0.16, perfect 0.1, extra 0.08, data 0.07, food 0.06, industry 0.06, happy 0.05, cancer 0.04, did 0.02, rich 0.02, say 0.02, want 0.02, kinds 0.01, 

Topic 13:
women 0.46, wave 0.18, woman 0.13, girls 0.12, extraordinary 0.12, worried 0.1, journey 0.09, happiness 0.08, met 0.07, violence 0.07, said 0.06, body 0.06, community 0.05, ice 0.05, began 0.04, 

Topic 14:
sound 0.27, wave 0.21, mr 0.16, design 0.14, technology 0.14, actually 0.13, changes 0.1, play 0.1, box 0.09, century 0.09, th 0.07, cool 0.07, mean 0.06, based 0.06, looks 0.06,
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
Top NMF topics in...
Al Gore 2006: 2 0 12
David Pogue 2006: 10 12 2
Cameron Sinclair 2006: 12 14 6
Jehane Noujaim 2006: 0 6 14
Larry Brilliant 2006: 9 2 14
Nicholas Negroponte 2006: 12 13 10
Jeff Han 2006: 13 10 1
Sirena Huang 2006: 1 11 2
Jennifer Lin 2006: 12 11 10
Amy Smith 2006: 13 12 11
Ross Lovegrove 2006: 13 12 10
Richard Baraniuk 2006: 12 0 6
Majora Carter 2006: 12 3 14
Jimmy Wales 2006: 12 0 11
Mena Trott 2006: 0 2 13
Ze Frank 2006: 0 11 2
Helen Fisher 2006: 4 14 13
Eve Ensler 2006: 14 1 11
David Deutsch 2006: 10 1 12
Richard Dawkins 2006: 6 1 4
Malcolm Gladwell 2006: 2 9 13
Steven Levitt 2006: 3 14 13
Barry Schwartz 2006: 11 10 13
Ken Robinson 2006: 2 13 0
Dan Gilbert 2006: 10 8 11
Eva Vertes 2006: 5 14 13
Aubrey de Grey 2006: 11 14 13
Iqbal Quadir 2006: 9 13 0
Jacqueline Novogratz 2006: 14 13 9
Ashraf Ghani 2006: 6 9 12
Sasa Vucinic 2006: 13 11 9
Burt Rutan 2006: 10 12 11
Ben Saunders 2006: 10 14 2
Edward Burtynsky 2006: 13 2 14
Hans Rosling 2006: 6 3 2
Robert Fischell 2006: 8 14 13
Bono 2006: 6 0 13
Michael Shermer 2006: 7 10 13
Peter Donnelly 2006: 7 14 13
Kevin Kelly 2006: 1 2 0
Ray Kurzweil 2006: 1 12 9
Peter Gabriel 2006: 6 0 10
Dean Ornish 2006: 9 8 12
Rives 2006: 10 0 8
Richard St. John 2006: 2 10 4
Tony Robbins 2006: 0 2 4
Joshua Prince-Ramus 2006: 12 1 2
Julia Sweeney 2006: 2 14 13
Rick Warren 2006: 2 0 1
Dan Dennett 2006: 2 1 8
```

```python
>>> doctopic.shape
(50, 15)
```

```python
>>> doctopic
array([[  5.47009445e-01,   6.42458320e-03,   6.74609594e-01,
          2.31820838e-01,   1.93108142e-01,   0.00000000e+00,
          4.15429304e-01,   1.04336530e-01,   0.00000000e+00,
          0.00000000e+00,   3.35002416e-01,   2.30887922e-01,
          4.88786866e-01,   1.25905629e-01,   1.80600026e-01],
       [  5.61154575e-01,   3.80024770e-01,   1.04059368e+00,
          0.00000000e+00,   1.72781091e-02,   0.00000000e+00,
          0.00000000e+00,   2.61249126e-01,   3.68440932e-01,
          0.00000000e+00,   2.40577850e+00,   7.23595678e-02,
          1.82774411e+00,   0.00000000e+00,   0.00000000e+00],
       [  5.50612669e-01,   0.00000000e+00,   7.14248427e-01,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          9.59162817e-01,   0.00000000e+00,   0.00000000e+00,
          1.21698675e-01,   0.00000000e+00,   0.00000000e+00,
          4.67371896e+00,   2.89336900e-01,   1.08063396e+00],
       [  5.49287115e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          9.83551849e-01,   0.00000000e+00,   0.00000000e+00,
          2.56210066e-02,   0.00000000e+00,   0.00000000e+00,
          1.03018544e-01,   4.45148699e-01,   8.09004681e-01],
       [  0.00000000e+00,   0.00000000e+00,   9.13677795e-02,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          9.02511425e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  2.95252294e-01,   0.00000000e+00,   5.24238607e-01,
          1.53082707e-01,   1.20213950e-01,   1.38484279e-02,
          9.18730140e-02,   2.50264107e-01,   9.84151974e-02,
          1.84084296e-01,   8.91312909e-01,   7.93085163e-01,
          1.48731481e+00,   9.48767642e-01,   0.00000000e+00],
       [  0.00000000e+00,   4.69321508e-01,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   3.78226884e-02,
          0.00000000e+00,   6.45003609e-02,   0.00000000e+00,
          0.00000000e+00,   5.32926341e-01,   3.12801172e-01,
          4.34399308e-01,   9.95497715e-01,   0.00000000e+00],
       [  0.00000000e+00,   5.44191989e-01,   1.86745064e-01,
          0.00000000e+00,   6.95501730e-02,   8.60503421e-03,
          0.00000000e+00,   0.00000000e+00,   4.54490431e-02,
          0.00000000e+00,   0.00000000e+00,   2.14498486e-01,
          1.50097701e-01,   0.00000000e+00,   1.65569544e-01],
       [  0.00000000e+00,   1.02357546e-02,   1.36743056e-01,
          0.00000000e+00,   7.25094963e-02,   9.24293390e-02,
          0.00000000e+00,   1.00039281e-01,   1.45119005e-01,
          0.00000000e+00,   3.42368195e-01,   4.11338280e-01,
          4.27810924e-01,   2.63038997e-02,   0.00000000e+00],
       [  3.36172128e-01,   3.01119453e-01,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   4.09616512e-02,
          2.61938316e-01,   0.00000000e+00,   1.13972274e-01,
          3.64662950e-01,   2.26771018e-01,   4.69774512e-01,
          1.01447003e+00,   1.59423247e+00,   2.88382222e-02],
       [  5.43231131e-01,   5.08941965e-01,   0.00000000e+00,
          0.00000000e+00,   4.39808513e-01,   2.02760828e-03,
          2.04883744e-01,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   1.26605131e+00,   1.17826735e-01,
          1.83081048e+00,   2.04502503e+00,   0.00000000e+00],
       [  7.80443487e-01,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          5.85720307e-01,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          5.65637104e+00,   0.00000000e+00,   0.00000000e+00],
       [  2.11048144e-01,   0.00000000e+00,   0.00000000e+00,
          5.89336357e-01,   0.00000000e+00,   0.00000000e+00,
          5.02733149e-01,   0.00000000e+00,   2.12856121e-01,
          3.70944539e-01,   0.00000000e+00,   0.00000000e+00,
          2.59916407e+00,   1.66316418e-01,   5.50374467e-01],
       [  1.61662989e+00,   0.00000000e+00,   5.85588187e-01,
          3.86668152e-01,   1.31791674e-01,   0.00000000e+00,
          0.00000000e+00,   2.09075231e-01,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   8.21909474e-01,
          3.56185110e+00,   0.00000000e+00,   0.00000000e+00],
       [  3.46308414e+00,   0.00000000e+00,   7.21544654e-01,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   7.37219847e-02,
          0.00000000e+00,   0.00000000e+00,   6.36904879e-01,
          0.00000000e+00,   6.64486454e-01,   1.66696618e-01],
       [  1.29594391e+00,   0.00000000e+00,   9.32858904e-01,
          1.69687144e-01,   1.67203349e-01,   0.00000000e+00,
          0.00000000e+00,   1.10912965e-01,   4.07201027e-02,
          0.00000000e+00,   3.10740570e-01,   1.24519016e+00,
          4.78478484e-01,   3.04546531e-01,   0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   1.05357834e+01,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   1.05478052e-05,   3.81576534e-01],
       [  0.00000000e+00,   1.69884118e-01,   0.00000000e+00,
          7.67778838e-02,   4.91345811e-02,   5.48825400e-03,
          0.00000000e+00,   2.24571473e-02,   0.00000000e+00,
          0.00000000e+00,   5.15674106e-02,   8.96885074e-02,
          0.00000000e+00,   0.00000000e+00,   9.70912823e+00],
       [  0.00000000e+00,   1.11499968e+00,   0.00000000e+00,
          0.00000000e+00,   1.90709234e-01,   0.00000000e+00,
          2.88199144e-01,   3.48549925e-01,   1.50916676e-01,
          0.00000000e+00,   2.06463569e+00,   2.31725758e-01,
          6.84259934e-01,   0.00000000e+00,   0.00000000e+00],
       [  3.11965805e-01,   9.17750586e-01,   0.00000000e+00,
          0.00000000e+00,   4.29017610e-01,   0.00000000e+00,
          3.67068550e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   3.10682380e-01,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  6.37968546e-01,   0.00000000e+00,   1.55310085e+00,
          4.60982591e-01,   3.43379458e-01,   4.65394275e-01,
          5.88102860e-02,   2.10061820e-01,   2.60422939e-01,
          1.21362450e+00,   0.00000000e+00,   8.32039811e-01,
          0.00000000e+00,   8.75183165e-01,   4.83401380e-02],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.02299835e+01,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  7.97002604e-01,   3.18164910e-01,   3.33179454e-01,
          4.00099455e-01,   8.52332790e-02,   0.00000000e+00,
          4.49997608e-01,   2.37356863e-01,   5.13418033e-01,
          3.66262725e-01,   8.77207785e-01,   9.26010236e-01,
          0.00000000e+00,   8.76114549e-01,   0.00000000e+00],
       [  7.85246704e-01,   3.60119027e-01,   2.39362072e+00,
          3.90259585e-02,   6.60321399e-01,   3.89099890e-02,
          0.00000000e+00,   1.76426342e-01,   1.50222423e-01,
          8.70362443e-02,   3.00808000e-01,   2.97144600e-01,
          5.70106048e-01,   7.99535592e-01,   4.47616506e-01],
       [  5.41582378e-01,   0.00000000e+00,   2.97809112e-01,
          3.88176044e-01,   9.78931089e-02,   0.00000000e+00,
          1.30927243e-01,   4.96245534e-01,   1.36501652e+00,
          7.44570366e-03,   1.98651624e+00,   8.86916254e-01,
          0.00000000e+00,   8.78356983e-01,   5.45705049e-01],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   9.31275479e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   1.18478626e+01,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  6.85778473e-01,   0.00000000e+00,   3.76984923e-02,
          5.21115074e-01,   0.00000000e+00,   0.00000000e+00,
          6.27767222e-01,   0.00000000e+00,   0.00000000e+00,
          1.00571064e+00,   1.88086274e-02,   4.14573884e-01,
          6.03062311e-01,   7.87164976e-01,   2.01322862e-01],
       [  4.51811289e-01,   0.00000000e+00,   0.00000000e+00,
          3.09650684e-01,   8.55554806e-02,   0.00000000e+00,
          6.87350628e-01,   0.00000000e+00,   1.60554569e-01,
          8.75478100e-01,   0.00000000e+00,   4.58926449e-01,
          2.75561254e-01,   8.92038440e-01,   1.18132820e+00],
       [  3.66540595e-01,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   3.29603251e-01,   0.00000000e+00,
          1.96995766e+00,   0.00000000e+00,   0.00000000e+00,
          6.83871260e-01,   0.00000000e+00,   3.46943479e-01,
          6.43294316e-01,   2.72275277e-01,   1.38387774e-01],
       [  0.00000000e+00,   2.57669017e-01,   3.33311047e-01,
          3.25114536e-01,   0.00000000e+00,   1.61416140e-01,
          3.12412164e-01,   4.23313884e-01,   3.38115330e-03,
          5.17756549e-01,   0.00000000e+00,   1.37537404e+00,
          3.51718431e-01,   1.74898492e+00,   0.00000000e+00],
       [  1.84335912e-01,   4.27566173e-02,   0.00000000e+00,
          1.97512441e-01,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   4.12285301e-01,
          3.83807373e-01,   3.83801689e+00,   9.04990609e-01,
          9.65610248e-01,   3.72244556e-01,   0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   2.86884195e-02,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   8.03730202e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   7.25107292e-02],
       [  0.00000000e+00,   0.00000000e+00,   1.08080328e-01,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   9.89340528e+00,   1.35473782e-02],
       [  0.00000000e+00,   0.00000000e+00,   1.09861132e-01,
          2.45749640e-01,   0.00000000e+00,   0.00000000e+00,
          6.47174997e+00,   2.55380592e-02,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   9.95662014e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   1.94517976e-02],
       [  1.08894018e+00,   4.21921798e-01,   7.00985530e-01,
          8.12004965e-02,   0.00000000e+00,   5.22686918e-02,
          2.97713425e+00,   1.78435252e-01,   4.09895434e-01,
          0.00000000e+00,   2.42016259e-01,   5.19360591e-01,
          1.16601519e-01,   7.66231695e-01,   0.00000000e+00],
       [  2.39924411e-01,   5.90224374e-02,   8.65562393e-02,
          3.22824053e-01,   1.23506388e-01,   6.92195424e-02,
          2.14659808e-01,   6.60470040e-01,   1.04076279e-01,
          0.00000000e+00,   6.25263875e-01,   5.09602537e-01,
          4.75588153e-01,   5.31973056e-01,   0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   1.10325336e+01,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   9.76535493e-02],
       [  8.88856376e-02,   8.25340761e+00,   2.44692878e-01,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  0.00000000e+00,   5.36453812e+00,   0.00000000e+00,
          0.00000000e+00,   3.10020308e-01,   3.79499565e-01,
          0.00000000e+00,   0.00000000e+00,   5.75730898e-01,
          1.23728610e+00,   0.00000000e+00,   0.00000000e+00,
          1.77371813e+00,   0.00000000e+00,   0.00000000e+00],
       [  5.88120960e-01,   1.31397186e-01,   7.05924493e-02,
          2.40923759e-01,   1.61675381e-01,   4.53268644e-02,
          9.23358229e-01,   1.49042011e-02,   0.00000000e+00,
          1.38557614e-01,   4.49353463e-01,   1.97770702e-01,
          0.00000000e+00,   2.09302838e-01,   3.13099772e-01],
       [  2.04585344e-01,   3.58793489e-04,   0.00000000e+00,
          8.83941071e-02,   0.00000000e+00,   1.85164321e-01,
          0.00000000e+00,   3.25088545e-02,   2.66984661e-01,
          3.02467849e-01,   0.00000000e+00,   1.37366986e-01,
          2.54472077e-01,   0.00000000e+00,   0.00000000e+00],
       [  1.44851731e-01,   1.16352670e-02,   4.71964636e-02,
          1.44597561e-02,   3.72813950e-03,   2.77812769e-02,
          6.61841492e-02,   0.00000000e+00,   7.73287795e-02,
          0.00000000e+00,   2.81242043e-01,   0.00000000e+00,
          2.87086268e-02,   1.72606682e-02,   0.00000000e+00],
       [  3.01198956e-02,   4.83249886e-02,   4.65362712e-01,
          1.68876066e-01,   1.84937095e-01,   0.00000000e+00,
          0.00000000e+00,   3.88386446e-02,   4.53759863e-02,
          0.00000000e+00,   3.60964242e-01,   6.34202681e-02,
          1.11492684e-01,   1.33861409e-01,   0.00000000e+00],
       [  4.53470583e+00,   9.04499420e-01,   1.70438695e+00,
          8.20237584e-02,   1.57190999e+00,   2.79848387e-02,
          0.00000000e+00,   5.42518952e-01,   0.00000000e+00,
          0.00000000e+00,   7.95221509e-01,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  0.00000000e+00,   8.38554541e-01,   7.18857498e-01,
          0.00000000e+00,   0.00000000e+00,   3.27274682e-02,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          4.06619200e+00,   3.91893625e-02,   0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   7.33757358e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   7.64664040e-01],
       [  1.43103591e+00,   1.01918583e+00,   3.52618483e+00,
          4.72069877e-01,   0.00000000e+00,   1.39689239e-01,
          2.33593094e-01,   0.00000000e+00,   6.74799017e-02,
          0.00000000e+00,   3.17293051e-01,   4.61133334e-01,
          0.00000000e+00,   3.81224351e-01,   0.00000000e+00],
       [  7.78120821e-01,   1.24173973e+00,   1.79141024e+00,
          0.00000000e+00,   6.03399722e-01,   1.70125956e-03,
          5.18370030e-01,   4.39490802e-02,   8.53848120e-01,
          9.95015124e-02,   0.00000000e+00,   6.18627384e-01,
          8.29240060e-01,   0.00000000e+00,   0.00000000e+00]])
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
/Users/john/Library/Python/3.4/lib/python/site-packages/matplotlib/axes/_axes.py:519: UserWarning: No labelled objects found. Use label='...' kwarg on individual plots.
  warnings.warn("No labelled objects found. "
```

## Working with LSI/LSA to get Elbow graph for k-means

I got the idea from [Sujit Pal][].

```python
>>> from sklearn.decomposition import TruncatedSVD
>>> from sklearn.preprocessing import Normalizer
...
>>> lsa = TruncatedSVD()
>>> my_lsa = lsa.fit_transform(tfidf)
>>> #normed = Normalizer(copy=False).fit_transform(my_lsa)
... #cosine_similarity(normed)
```

```python
>>> print(normed)
```

```python
>>> # write out coordinates to file
... fcoords = open(os.path.join(MODELS_DIR, "coords.csv"), 'wb')
>>> for vector in lsi[corpus]:
...     if len(vector) != 2:
...         continue
...     fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
>>> fcoords.close()
```
