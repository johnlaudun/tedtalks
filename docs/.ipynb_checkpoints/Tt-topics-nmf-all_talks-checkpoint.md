# TEDtalks: `pandas` filters and visualizations

```python
>>> %pylab inline
Populating the interactive namespace from numpy and matplotlib
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
>>> n_topics = 45
>>> n_top_words = 15
>>> tt_stopwords = open('../data/tt_stopwords.txt', 'r').read().splitlines()
...
...
>>> # Use tf-idf features for NMF.
... tfidf_vectorizer = text.TfidfVectorizer(max_df = 0.95,
...                                         min_df = 2,
...                                         max_features = n_features,
...                                         stop_words = tt_stopwords)
...
>>> tfidf = tfidf_vectorizer.fit_transform(utalks)
...
>>> # I don't think we need this. (Duplicate.)
... # tf_vectorizer = text.CountVectorizer(max_df=0.95,
... #                                      min_df=2,
... #                                      max_features=n_features,
... #                                      stop_words='english')
...
... # Fit the NMF model
... print("Fitting the NMF model with {} topics for {} documents with {} features."
...       .format(n_topics, n_samples, n_features))
>>> nmf = NMF(n_components=n_topics,
...           random_state=1,
...           alpha=.1,
...           l1_ratio=.5).fit(tfidf)
Fitting the NMF model with 45 topics for 2113 documents with 1000 features.
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

Topics in NMF model:

Topic 0:
just 0.93, really 0.91, actually 0.9, going 0.83, see 0.79, think 0.72, things 0.7, get 0.67, time 0.56, make 0.53, little 0.52, right 0.5, thing 0.5, look 0.5, go 0.5, 

Topic 1:
women 2.79, men 1.12, woman 0.46, sex 0.29, female 0.28, violence 0.15, rights 0.11, girls 0.1, young 0.1, talk 0.09, stories 0.09, media 0.09, country 0.08, education 0.08, percent 0.07, 

Topic 2:
percent 0.65, countries 0.6, world 0.6, dollars 0.57, money 0.49, global 0.47, billion 0.42, economy 0.41, business 0.41, growth 0.4, country 0.39, economic 0.39, companies 0.37, market 0.36, people 0.31, 

Topic 3:
ocean 1.65, fish 1.06, sea 0.88, deep 0.24, blue 0.17, water 0.14, surface 0.14, north 0.13, species 0.13, planet 0.11, area 0.11, places 0.11, miles 0.11, areas 0.11, land 0.1, 

Topic 4:
cells 2.27, cell 0.78, body 0.29, disease 0.2, drug 0.19, lab 0.18, blood 0.16, skin 0.13, diseases 0.13, drugs 0.12, grow 0.12, heart 0.11, actually 0.11, material 0.1, patient 0.1, 

Topic 5:
universe 1.99, space 0.44, theory 0.4, stars 0.37, physics 0.34, dark 0.28, billion 0.2, science 0.19, black 0.18, matter 0.17, energy 0.16, mass 0.14, big 0.14, earth 0.12, field 0.12, 

Topic 6:
music 2.57, play 0.31, playing 0.24, piece 0.22, song 0.22, hear 0.19, listen 0.1, video 0.09, thank 0.08, experience 0.07, heard 0.06, audience 0.06, kind 0.05, beautiful 0.05, traditional 0.05, 

Topic 7:
city 2.05, cities 1.36, new 0.33, map 0.32, york 0.32, street 0.24, park 0.22, public 0.17, community 0.17, people 0.16, places 0.15, citizens 0.13, site 0.13, live 0.13, cars 0.12, 

Topic 8:
brain 2.82, brains 0.39, activity 0.18, mental 0.17, human 0.17, body 0.16, memory 0.14, mind 0.14, behavior 0.13, visual 0.12, pain 0.11, control 0.11, genes 0.1, normal 0.1, study 0.1, 

Topic 9:
africa 2.17, african 0.84, hiv 0.46, countries 0.3, south 0.29, aid 0.19, leaders 0.16, world 0.12, country 0.12, east 0.1, west 0.07, story 0.07, people 0.06, traditional 0.06, map 0.06, 

Topic 10:
cancer 2.46, body 0.19, disease 0.19, drug 0.18, blood 0.15, gene 0.09, treatment 0.09, lab 0.07, percent 0.07, genetic 0.06, patient 0.05, genes 0.05, research 0.05, found 0.05, cell 0.05, 

Topic 11:
robot 1.89, robots 1.21, move 0.1, video 0.09, see 0.07, lab 0.07, build 0.06, control 0.06, foot 0.05, body 0.05, machines 0.04, animal 0.04, small 0.04, intelligence 0.04, moving 0.03, 

Topic 12:
kids 1.5, school 1.17, students 0.95, education 0.71, teachers 0.7, teacher 0.4, learning 0.35, schools 0.34, teach 0.28, class 0.27, kid 0.26, student 0.24, learn 0.22, math 0.19, parents 0.15, 

Topic 13:
data 2.83, information 0.51, web 0.18, map 0.17, points 0.1, numbers 0.1, patterns 0.08, google 0.08, see 0.07, look 0.07, science 0.06, percent 0.05, social 0.05, online 0.05, rate 0.05, 

Topic 14:
design 2.55, designed 0.3, products 0.16, product 0.16, beautiful 0.13, work 0.12, materials 0.11, object 0.11, new 0.11, process 0.1, made 0.08, kind 0.08, really 0.07, ideas 0.07, working 0.06, 

Topic 15:
dna 1.75, bacteria 0.63, genes 0.55, gene 0.51, genetic 0.42, species 0.29, cell 0.25, code 0.24, biology 0.17, evolution 0.13, science 0.1, years 0.1, human 0.08, understand 0.08, bodies 0.07, 

Topic 16:
patients 1.09, health 0.91, patient 0.75, disease 0.66, care 0.57, medical 0.51, drugs 0.44, doctors 0.44, hiv 0.42, treatment 0.39, medicine 0.38, drug 0.38, doctor 0.36, hospital 0.35, heart 0.33, 

Topic 17:
girls 1.99, girl 0.77, school 0.31, father 0.12, boy 0.09, wish 0.08, village 0.08, am 0.05, parents 0.04, research 0.04, young 0.04, community 0.04, code 0.03, change 0.03, know 0.03, 

Topic 18:
chinese 1.49, china 1.32, india 0.31, west 0.15, world 0.14, north 0.13, countries 0.11, east 0.11, state 0.09, political 0.08, states 0.08, united 0.08, western 0.06, american 0.06, growth 0.06, 

Topic 19:
food 2.23, eat 0.67, feed 0.3, waste 0.25, plant 0.23, plants 0.17, fish 0.16, healthy 0.14, kids 0.13, local 0.11, growing 0.11, know 0.09, grow 0.09, produce 0.09, really 0.07, 

Topic 20:
know 1.71, going 0.73, yeah 0.68, just 0.68, right 0.55, oh 0.55, ok 0.54, say 0.5, think 0.44, get 0.36, love 0.36, thing 0.35, really 0.34, mean 0.33, guy 0.33, 

Topic 21:
compassion 1.86, god 0.41, world 0.12, beings 0.12, happy 0.11, human 0.11, love 0.1, says 0.08, self 0.08, happiness 0.07, person 0.07, word 0.06, zero 0.06, feel 0.05, good 0.05, 

Topic 22:
animals 1.99, animal 0.89, species 0.6, human 0.33, humans 0.33, nature 0.24, female 0.21, fish 0.18, sex 0.15, planet 0.13, behavior 0.11, evolution 0.11, skin 0.09, found 0.08, eat 0.08, 

Topic 23:
play 1.31, game 1.31, games 1.14, video 0.55, playing 0.38, fun 0.1, win 0.1, physical 0.08, thank 0.07, real 0.07, world 0.07, serious 0.06, social 0.05, activity 0.05, online 0.05, 

Topic 24:
light 2.12, see 0.6, dark 0.33, stars 0.28, eyes 0.28, blue 0.26, eye 0.26, camera 0.22, sun 0.22, night 0.19, color 0.17, black 0.16, image 0.16, video 0.15, deep 0.14, 

Topic 25:
water 2.75, river 0.33, waste 0.23, bacteria 0.17, surface 0.12, material 0.11, environment 0.11, clean 0.11, air 0.09, india 0.09, green 0.07, south 0.07, body 0.06, sea 0.06, produce 0.05, 

Topic 26:
forest 1.0, plants 0.98, plant 0.79, trees 0.76, species 0.66, tree 0.43, carbon 0.32, climate 0.19, land 0.18, nature 0.18, grow 0.17, area 0.14, world 0.14, percent 0.12, green 0.11, 

Topic 27:
sound 1.94, voice 0.68, listening 0.51, sounds 0.43, hear 0.34, song 0.27, listen 0.23, play 0.11, color 0.11, visual 0.1, just 0.08, body 0.07, time 0.07, health 0.06, video 0.06, 

Topic 28:
children 2.21, child 0.97, parents 0.37, babies 0.35, india 0.3, family 0.29, families 0.27, mother 0.18, schools 0.17, boy 0.14, baby 0.14, education 0.11, countries 0.1, school 0.09, development 0.08, 

Topic 29:
people 3.39, think 0.54, social 0.3, person 0.28, things 0.26, money 0.25, group 0.24, moral 0.23, say 0.22, somebody 0.2, change 0.19, lot 0.18, percent 0.17, know 0.15, study 0.15, 

Topic 30:
energy 1.74, oil 1.02, power 0.47, nuclear 0.47, solar 0.42, electricity 0.35, gas 0.35, carbon 0.31, wind 0.3, climate 0.28, air 0.16, stuff 0.13, technology 0.12, co 0.11, need 0.11, 

Topic 31:
car 1.79, cars 1.16, driving 0.41, road 0.33, drive 0.29, miles 0.23, cities 0.12, hour 0.11, going 0.1, cost 0.07, expensive 0.07, police 0.07, percent 0.07, stop 0.06, gas 0.05, 

Topic 32:
technology 1.38, information 1.01, digital 0.78, phone 0.63, computer 0.56, mobile 0.51, technologies 0.46, device 0.37, world 0.31, computers 0.31, physical 0.27, software 0.26, tools 0.24, system 0.23, human 0.21, 

Topic 33:
machine 2.0, machines 0.62, computer 0.58, computers 0.3, human 0.13, memory 0.12, learning 0.11, web 0.1, built 0.1, intelligence 0.09, build 0.09, hospital 0.07, running 0.06, read 0.04, robots 0.04, 

Topic 34:
ice 1.74, climate 0.57, feet 0.2, sea 0.18, north 0.16, south 0.15, change 0.15, co 0.15, planet 0.12, years 0.12, carbon 0.11, cold 0.11, year 0.11, global 0.1, ocean 0.1, 

Topic 35:
life 0.74, day 0.49, story 0.47, man 0.47, time 0.46, years 0.44, father 0.42, love 0.41, family 0.39, home 0.38, mother 0.36, old 0.34, stories 0.34, people 0.33, first 0.33, 

Topic 36:
earth 1.27, planet 1.03, mars 0.97, solar 0.52, life 0.45, sun 0.41, stars 0.38, surface 0.35, space 0.32, system 0.24, years 0.18, miles 0.16, science 0.12, human 0.12, billion 0.12, 

Topic 37:
said 2.77, went 0.35, god 0.23, didn 0.22, asked 0.2, thought 0.2, came 0.2, going 0.19, say 0.18, looked 0.17, come 0.12, ll 0.12, oh 0.1, guy 0.1, mom 0.1, 

Topic 38:
language 1.63, english 1.1, word 0.63, words 0.59, speak 0.22, say 0.16, writing 0.15, write 0.13, read 0.11, books 0.11, learn 0.11, learning 0.09, meaning 0.08, computer 0.08, speaking 0.08, 

Topic 39:
art 1.56, book 0.88, books 0.61, images 0.53, image 0.35, stories 0.29, read 0.25, film 0.25, story 0.24, work 0.23, project 0.21, visual 0.19, reading 0.16, world 0.15, paper 0.14, 

Topic 40:
building 1.42, buildings 0.82, architecture 0.79, space 0.64, project 0.35, built 0.32, build 0.29, materials 0.23, air 0.23, house 0.21, site 0.21, structure 0.18, material 0.17, public 0.17, center 0.15, 

Topic 41:
democracy 1.22, government 0.81, political 0.73, citizens 0.46, politics 0.42, rights 0.4, power 0.35, country 0.27, society 0.21, freedom 0.19, public 0.18, states 0.18, governments 0.17, revolution 0.16, values 0.15, 

Topic 42:
war 1.57, peace 0.68, violence 0.62, conflict 0.57, military 0.35, security 0.31, killed 0.3, international 0.27, states 0.18, force 0.16, world 0.15, groups 0.15, news 0.15, east 0.14, region 0.13, 

Topic 43:
internet 1.58, media 0.78, web 0.71, online 0.64, page 0.39, google 0.27, content 0.27, social 0.26, video 0.25, network 0.24, companies 0.17, news 0.16, government 0.16, website 0.14, message 0.12, 

Topic 44:
laughter 2.48, applause 0.5, okay 0.22, piece 0.09, oh 0.09, friends 0.08, color 0.07, thank 0.07, mom 0.07, right 0.07, four 0.05, science 0.05, open 0.04, yeah 0.04, local 0.04,
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
```

```python
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
>>> plt.figure(figsize=(144,12))
>>> fig = matplotlib.pyplot.gcf()
>>> fig.set_size_inches(144, 12)
...
>>> N, K = doctopic.shape  # N documents, K topics
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
>>> plt.title('Topics in TEDtalks')
>>> plt.xticks(ind+width/2, citations, rotation='vertical', fontsize=6)
>>> plt.yticks(np.arange(0, 1, 10))
>>> topic_labels = ['Topic #{}'.format(k) for k in range(K)]
>>> plt.legend([p[0] for p in plots], topic_labels)
>>> plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
>>> plt.show()
/Users/john/Library/Python/3.4/lib/python/site-packages/matplotlib/axes/_axes.py:519: UserWarning: No labelled objects found. Use label='...' kwarg on individual plots.
  warnings.warn("No labelled objects found. "
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
