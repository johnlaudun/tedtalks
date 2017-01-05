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
>>> data2016 = df[df['date'] == '2016']
...
>>> talks = data2016.text.tolist()
>>> authors = data2016.author.tolist()
>>> dates = data2016.date.tolist()
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
>>> n_topics = 45
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
... # tf = tf_vectorizer.fit_transform(talks)
```

```python
>>> # Fit the NMF model
... print("Fitting the NMF model with tf-idf features, "
...       "n_topics={}, n_samples={} and n_features={}...".format(n_topics, n_samples, n_features))
>>> nmf = NMF(n_components=n_topics,
...           random_state=1,
...           alpha=.1,
...           l1_ratio=.5).fit(tfidf)
Fitting the NMF model with tf-idf features, n_topics=45, n_samples=95 and n_features=1000...
```

```python
>>> print("\nTopics in NMF model:")
>>> tfidf_feature_names = tfidf_vectorizer.get_feature_names()
>>> print_top_words(nmf, tfidf_feature_names, n_top_words) #n_top_words can be changed on the fly

Topics in NMF model:

Topic 0:
ca 0.37, source 0.24, code 0.24, people 0.18, open 0.16, mean 0.12, really 0.11, project 0.09, software 0.08, kind 0.06, point 0.05, big 0.03, actually 0.03, interesting 0.03, nice 0.02, 

Topic 1:
mars 0.85, earth 0.2, atmosphere 0.12, planet 0.1, humans 0.09, space 0.09, biology 0.06, species 0.05, human 0.05, water 0.04, ice 0.04, survive 0.02, technology 0.02, reality 0.02, ways 0.02, 

Topic 2:
young 0.0, fix 0.0, friend 0.0, free 0.0, fossil 0.0, forward 0.0, forms 0.0, form 0.0, forget 0.0, forever 0.0, force 0.0, food 0.0, focus 0.0, flying 0.0, fly 0.0, 

Topic 3:
plant 0.51, plants 0.4, genes 0.31, food 0.28, water 0.19, drought 0.15, communities 0.08, climate 0.06, percent 0.04, known 0.04, environment 0.04, platform 0.04, approach 0.04, data 0.03, weeks 0.03, 

Topic 4:
brain 0.84, cells 0.48, brains 0.16, story 0.12, listeners 0.1, monkey 0.07, similar 0.07, dreams 0.07, patients 0.07, sound 0.03, bg 0.03, cell 0.03, exactly 0.03, inside 0.03, ability 0.03, 

Topic 5:
race 0.56, patients 0.25, black 0.22, medicine 0.19, doctors 0.17, health 0.16, medical 0.13, genetic 0.12, drug 0.11, white 0.1, drugs 0.09, social 0.09, patient 0.06, african 0.04, woman 0.03, 

Topic 6:
water 0.82, supply 0.13, cities 0.1, treatment 0.08, city 0.08, plant 0.08, urban 0.07, communities 0.05, process 0.04, solve 0.04, san 0.03, step 0.02, drought 0.02, local 0.01, effects 0.01, 

Topic 7:
young 0.0, fix 0.0, friend 0.0, free 0.0, fossil 0.0, forward 0.0, forms 0.0, form 0.0, forget 0.0, forever 0.0, force 0.0, food 0.0, focus 0.0, flying 0.0, fly 0.0, 

Topic 8:
girls 0.73, periods 0.45, women 0.19, book 0.09, code 0.08, teach 0.08, parents 0.05, india 0.03, stories 0.02, girl 0.02, period 0.02, percent 0.02, perfect 0.02, taught 0.02, young 0.01, 

Topic 9:
data 0.58, credit 0.27, shows 0.2, tv 0.15, decision 0.14, amazon 0.12, points 0.12, analysis 0.11, pieces 0.11, curve 0.1, decisions 0.07, price 0.07, financial 0.05, business 0.05, bank 0.04, 

Topic 10:
young 0.0, fix 0.0, friend 0.0, free 0.0, fossil 0.0, forward 0.0, forms 0.0, form 0.0, forget 0.0, forever 0.0, force 0.0, food 0.0, focus 0.0, flying 0.0, fly 0.0, 

Topic 11:
young 0.0, fix 0.0, friend 0.0, free 0.0, fossil 0.0, forward 0.0, forms 0.0, form 0.0, forget 0.0, forever 0.0, force 0.0, food 0.0, focus 0.0, flying 0.0, fly 0.0, 

Topic 12:
universe 0.5, waves 0.3, higgs 0.28, theory 0.19, physics 0.13, field 0.11, energy 0.11, mass 0.08, space 0.08, black 0.07, stars 0.06, sound 0.06, dark 0.05, exist 0.05, fine 0.04, 

Topic 13:
christopher 0.47, criminal 0.44, justice 0.22, law 0.09, decisions 0.08, school 0.05, money 0.04, police 0.03, job 0.03, young 0.02, internship 0.02, record 0.02, spend 0.02, community 0.02, public 0.02, 

Topic 14:
economic 0.36, capitalism 0.35, democracy 0.32, growth 0.26, sphere 0.2, bg 0.15, political 0.15, countries 0.11, china 0.09, continue 0.05, capital 0.05, free 0.05, market 0.04, mountain 0.04, government 0.04, 

Topic 15:
young 0.0, fix 0.0, friend 0.0, free 0.0, fossil 0.0, forward 0.0, forms 0.0, form 0.0, forget 0.0, forever 0.0, force 0.0, food 0.0, focus 0.0, flying 0.0, fly 0.0, 

Topic 16:
cancer 0.56, tumor 0.39, cell 0.17, drug 0.15, gene 0.08, signal 0.08, size 0.07, cells 0.07, genetic 0.06, engineering 0.06, body 0.06, blood 0.06, core 0.05, detect 0.05, patients 0.04, 

Topic 17:
design 0.75, designers 0.33, trust 0.09, teach 0.06, phone 0.05, course 0.05, powerful 0.04, patients 0.04, designed 0.04, stay 0.02, sharing 0.02, plastic 0.02, different 0.02, projects 0.01, internet 0.01, 

Topic 18:
corals 0.48, ocean 0.42, coral 0.24, sharks 0.21, underwater 0.14, sea 0.13, blue 0.13, deep 0.12, meters 0.12, fish 0.08, ve 0.08, green 0.06, light 0.06, beautiful 0.05, swimming 0.05, 

Topic 19:
rock 0.14, students 0.13, groups 0.12, solve 0.12, friends 0.12, step 0.11, creative 0.09, try 0.09, performance 0.08, list 0.08, house 0.07, worked 0.07, play 0.07, complicated 0.06, difficult 0.05, 

Topic 20:
connectivity 0.37, cities 0.25, world 0.12, china 0.12, map 0.11, countries 0.1, network 0.1, global 0.1, borders 0.09, war 0.07, india 0.07, trade 0.07, million 0.06, billion 0.06, dollars 0.05, 

Topic 21:
fish 0.5, healthy 0.2, farming 0.19, feed 0.15, oceans 0.12, animal 0.12, ocean 0.1, food 0.07, global 0.07, planet 0.07, need 0.07, tons 0.06, eat 0.06, industry 0.05, needs 0.05, 

Topic 22:
olympics 0.46, intellectual 0.35, disabilities 0.33, special 0.29, games 0.13, world 0.04, people 0.03, hidden 0.02, health 0.02, word 0.02, million 0.01, year 0.01, half 0.0, team 0.0, friends 0.0, 

Topic 23:
government 0.52, services 0.23, team 0.15, service 0.12, digital 0.12, united 0.06, states 0.06, projects 0.05, application 0.04, easy 0.04, work 0.03, president 0.03, change 0.03, inside 0.03, online 0.03, 

Topic 24:
young 0.0, fix 0.0, friend 0.0, free 0.0, fossil 0.0, forward 0.0, forms 0.0, form 0.0, forget 0.0, forever 0.0, force 0.0, food 0.0, focus 0.0, flying 0.0, fly 0.0, 

Topic 25:
said 0.57, email 0.34, gold 0.26, drugs 0.07, code 0.05, day 0.04, business 0.04, going 0.04, doing 0.04, start 0.03, bank 0.03, highly 0.03, worth 0.03, spending 0.02, thought 0.02, 

Topic 26:
ln 0.67, bone 0.18, desert 0.12, north 0.07, story 0.06, totally 0.06, species 0.05, long 0.03, yeah 0.03, scientists 0.03, turns 0.03, tiny 0.02, hot 0.02, different 0.02, million 0.02, 

Topic 27:
dreams 0.48, dream 0.21, moon 0.18, humanity 0.15, fear 0.08, picture 0.08, everybody 0.08, talking 0.07, set 0.05, potential 0.05, simply 0.05, help 0.04, right 0.04, technology 0.04, thomas 0.04, 

Topic 28:
river 0.67, hot 0.33, water 0.09, coffee 0.09, amazon 0.06, gold 0.05, cold 0.05, unique 0.03, earth 0.03, degrees 0.02, heat 0.02, extra 0.02, waves 0.01, asked 0.01, exist 0.0, 

Topic 29:
children 0.33, lying 0.24, lies 0.23, lie 0.2, detect 0.19, emotions 0.18, child 0.16, hidden 0.1, blood 0.08, truth 0.07, tell 0.05, reading 0.04, technology 0.04, changes 0.04, percent 0.04, 

Topic 30:
age 0.25, pages 0.24, wrote 0.24, list 0.21, life 0.2, year 0.12, old 0.08, book 0.07, sure 0.07, read 0.06, patterns 0.06, clear 0.06, die 0.06, shared 0.05, live 0.05, 

Topic 31:
patents 0.32, myriad 0.28, gene 0.24, court 0.2, dna 0.17, chris 0.13, genes 0.12, gold 0.11, case 0.09, test 0.09, cancer 0.06, nature 0.05, result 0.04, issue 0.04, testing 0.04, 

Topic 32:
ebola 0.49, fight 0.15, data 0.1, health 0.09, humanity 0.08, individuals 0.08, work 0.08, worked 0.07, team 0.07, came 0.07, release 0.05, capacity 0.05, cases 0.05, extraordinary 0.05, lives 0.04, 

Topic 33:
people 0.27, world 0.2, going 0.18, time 0.18, think 0.17, ve 0.16, know 0.15, don 0.14, new 0.14, make 0.14, really 0.14, right 0.13, years 0.13, things 0.13, want 0.12, 

Topic 34:
gene 0.68, drive 0.24, drives 0.21, species 0.17, genes 0.15, release 0.07, basically 0.05, fly 0.04, spread 0.04, tool 0.03, red 0.02, engineer 0.02, ll 0.01, population 0.01, theory 0.01, 

Topic 35:
flying 0.33, machines 0.25, fly 0.23, machine 0.17, moving 0.13, wants 0.13, space 0.12, technology 0.08, parts 0.07, example 0.07, highly 0.05, performance 0.05, allows 0.05, new 0.04, developed 0.04, 

Topic 36:
internship 0.45, career 0.38, break 0.18, work 0.13, programs 0.13, job 0.12, return 0.07, program 0.06, loved 0.04, company 0.03, companies 0.03, business 0.03, tell 0.02, told 0.02, touch 0.02, 

Topic 37:
rocks 0.48, dinosaurs 0.41, rock 0.14, earth 0.13, bone 0.1, desert 0.07, record 0.05, age 0.05, fossil 0.05, planet 0.05, pages 0.04, species 0.04, place 0.03, giant 0.03, river 0.02, 

Topic 38:
star 0.38, planet 0.38, planets 0.3, stars 0.24, light 0.23, ice 0.14, climate 0.08, search 0.07, surface 0.07, earth 0.07, life 0.06, blue 0.06, atmosphere 0.06, models 0.05, data 0.05, 

Topic 39:
conversation 0.31, listen 0.17, listening 0.16, don 0.15, number 0.15, conversations 0.12, talk 0.09, talking 0.08, said 0.08, attention 0.07, learn 0.06, host 0.05, mind 0.05, going 0.05, means 0.04, 

Topic 40:
media 0.32, social 0.22, page 0.2, internet 0.15, people 0.12, conversations 0.12, write 0.09, revolution 0.08, facebook 0.07, minds 0.06, online 0.06, days 0.06, experiences 0.05, behavior 0.05, spread 0.05, 

Topic 41:
poem 0.52, human 0.29, written 0.23, computer 0.21, test 0.18, write 0.09, computers 0.05, language 0.05, asking 0.05, text 0.03, material 0.03, think 0.02, ok 0.02, ve 0.02, answer 0.02, 

Topic 42:
stuff 0.29, random 0.23, natural 0.16, word 0.14, baby 0.13, changing 0.1, animals 0.09, doing 0.08, plants 0.07, thinking 0.07, choose 0.07, movie 0.06, technologies 0.05, forms 0.05, humans 0.05, 

Topic 43:
global 0.4, climate 0.35, governments 0.21, printing 0.21, money 0.2, citizens 0.17, change 0.16, countries 0.15, dollars 0.12, national 0.11, carbon 0.1, extra 0.1, fund 0.1, financial 0.09, crisis 0.08, 

Topic 44:
computer 0.33, computers 0.28, kids 0.2, world 0.13, really 0.1, technology 0.09, girl 0.09, learn 0.08, favorite 0.07, teach 0.07, built 0.07, little 0.07, boy 0.05, smaller 0.04, parents 0.04,
```

```python
>>> # Now to associate NMF topics to documents...
... tf = tf_vectorizer.fit_transform(talks)
>>> dtm = tf.toarray()
>>> doctopic = nmf.fit_transform(dtm)
>>> print("Top NMF topics in...")
>>> for i in range(len(doctopic)):
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
>>> plt.figure(figsize=(24,8))
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
>>> plt.title('Topics in 2016 TEDtalks')
>>> plt.xticks(ind+width/2, citations, rotation='vertical')
>>> plt.yticks(np.arange(0, 1, 10))
>>> topic_labels = ['Topic #{}'.format(k) for k in range(K)]
>>> #plt.legend([p[0] for p in plots], topic_labels)
... #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
... plt.show()
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
