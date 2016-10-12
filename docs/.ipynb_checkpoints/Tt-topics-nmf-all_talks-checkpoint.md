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

Topics in NMF model:

Topic 0:
actually 0.93, really 0.89, just 0.88, think 0.7, things 0.68, going 0.65, way 0.59, time 0.58, ve 0.57, make 0.52, 

Topic 1:
said 0.7, life 0.67, story 0.51, day 0.47, man 0.45, time 0.43, father 0.43, years 0.43, family 0.39, went 0.39, 

Topic 2:
world 0.67, countries 0.62, country 0.5, global 0.47, percent 0.47, government 0.46, china 0.42, dollars 0.4, economic 0.39, economy 0.37, 

Topic 3:
patients 1.11, health 0.98, patient 0.8, disease 0.63, care 0.6, medical 0.54, doctors 0.45, medicine 0.38, doctor 0.37, hospital 0.37, 

Topic 4:
universe 1.96, stars 0.44, space 0.44, light 0.4, theory 0.37, dark 0.37, physics 0.33, black 0.23, billion 0.21, energy 0.19, 

Topic 5:
women 2.74, men 1.1, girls 0.49, woman 0.46, female 0.28, sex 0.28, violence 0.18, girl 0.17, rights 0.12, young 0.11, 

Topic 6:
cancer 2.46, drug 0.23, disease 0.2, body 0.19, blood 0.15, treatment 0.11, gene 0.09, percent 0.07, lab 0.07, drugs 0.07, 

Topic 7:
brain 2.82, brains 0.38, activity 0.19, mental 0.17, human 0.17, body 0.16, memory 0.14, mind 0.13, behavior 0.13, visual 0.11, 

Topic 8:
building 1.42, buildings 0.81, architecture 0.78, space 0.65, project 0.36, built 0.31, build 0.28, air 0.25, site 0.22, house 0.22, 

Topic 9:
ca 2.2, yeah 0.23, mean 0.18, chris 0.15, think 0.12, cause 0.1, got 0.09, ted 0.06, yes 0.05, ok 0.04, 

Topic 10:
kids 1.32, school 1.2, children 0.97, students 0.77, education 0.71, teachers 0.67, schools 0.42, child 0.37, teacher 0.37, learning 0.35, 

Topic 11:
robot 1.89, robots 1.2, legs 0.12, video 0.09, lab 0.07, animal 0.06, build 0.06, control 0.06, foot 0.05, want 0.05, 

Topic 12:
music 2.57, play 0.3, playing 0.24, song 0.23, piece 0.22, hear 0.18, listen 0.1, thank 0.09, video 0.08, experience 0.07, 

Topic 13:
ocean 1.6, fish 1.0, sea 0.88, animals 0.3, deep 0.27, ice 0.25, blue 0.19, light 0.17, north 0.17, water 0.15, 

Topic 14:
cells 2.29, cell 0.78, body 0.28, disease 0.21, drug 0.18, lab 0.17, blood 0.16, skin 0.14, diseases 0.13, heart 0.12, 

Topic 15:
city 2.01, cities 1.37, map 0.32, new 0.31, york 0.31, cars 0.28, street 0.25, park 0.22, car 0.18, public 0.17, 

Topic 16:
play 1.38, game 1.28, games 1.11, video 0.52, playing 0.38, win 0.1, fun 0.1, thank 0.09, physical 0.09, world 0.07, 

Topic 17:
language 1.58, english 1.08, word 0.61, words 0.58, speak 0.22, chinese 0.21, books 0.17, say 0.16, writing 0.15, example 0.14, 

Topic 18:
water 2.7, river 0.32, waste 0.23, bacteria 0.17, ice 0.13, use 0.12, surface 0.11, clean 0.11, ve 0.11, material 0.11, 

Topic 19:
africa 2.2, african 0.88, south 0.28, aid 0.21, countries 0.21, leaders 0.15, country 0.09, world 0.08, market 0.08, east 0.07, 

Topic 20:
people 3.39, think 0.43, money 0.32, social 0.29, person 0.28, don 0.28, percent 0.25, want 0.25, things 0.24, group 0.24, 

Topic 21:
animals 1.12, species 1.05, forest 0.86, plants 0.67, trees 0.61, animal 0.43, tree 0.42, plant 0.41, nature 0.33, human 0.27, 

Topic 22:
hiv 1.69, drugs 0.63, drug 0.44, treatment 0.29, sex 0.22, countries 0.2, positive 0.11, developing 0.07, available 0.06, disease 0.06, 

Topic 23:
data 2.84, web 0.17, map 0.16, information 0.12, points 0.11, numbers 0.1, look 0.09, text 0.08, patterns 0.08, using 0.07, 

Topic 24:
compassion 1.84, god 0.42, beings 0.11, world 0.11, human 0.11, happy 0.11, love 0.1, says 0.09, self 0.08, happiness 0.07, 

Topic 25:
earth 1.23, planet 1.0, mars 0.9, solar 0.5, ice 0.47, sun 0.43, life 0.41, stars 0.4, surface 0.35, star 0.28, 

Topic 26:
energy 1.64, oil 1.0, nuclear 0.4, climate 0.39, solar 0.38, carbon 0.36, power 0.36, electricity 0.35, gas 0.34, wind 0.28, 

Topic 27:
sound 1.93, voice 0.67, listening 0.5, sounds 0.44, hear 0.35, song 0.27, listen 0.23, color 0.12, laughter 0.12, play 0.11, 

Topic 28:
know 1.43, going 1.01, said 0.9, don 0.85, just 0.7, got 0.68, right 0.57, say 0.56, oh 0.52, want 0.51, 

Topic 29:
food 2.17, eat 0.66, plant 0.4, plants 0.31, feed 0.3, waste 0.25, fish 0.18, healthy 0.14, kids 0.13, growing 0.12, 

Topic 30:
machine 1.85, computer 0.97, machines 0.62, computers 0.47, technology 0.24, human 0.21, intelligence 0.15, learning 0.14, built 0.12, memory 0.11, 

Topic 31:
dna 1.78, bacteria 0.61, genes 0.54, gene 0.51, genetic 0.41, cell 0.25, code 0.24, species 0.23, biology 0.17, evolution 0.12, 

Topic 32:
internet 1.52, media 0.73, web 0.65, online 0.63, page 0.36, digital 0.36, government 0.31, video 0.31, content 0.29, network 0.26, 

Topic 33:
design 2.57, designed 0.3, products 0.15, product 0.15, beautiful 0.12, art 0.12, work 0.11, technology 0.1, materials 0.1, new 0.1, 

Topic 34:
information 2.39, phone 0.43, mobile 0.28, visual 0.21, intelligence 0.11, access 0.11, police 0.1, digital 0.09, device 0.06, technologies 0.06,
```

```python
>>> # function for saving topic list to dataframe
... for topic_id, topic in enumerate(nmf.components_):
...     ids = []
...     keywords = []
...     ids.append('\nTopic {}'.format(int(topic_id)))
...     keywords. append(''.join([features[i] + ' ' + str(round(topic[i], 2))
...           +', ' for i in topic.argsort()[:-n_top_words - 1:-1]]))
...
>>> topics_df = pandas.DataFrame(
...     {'TopicID': ids,
...      'Keywords': keywords
...     })
```

```python

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
Top NMF topics in...
Al Gore 2006: 26 2 1
David Pogue 2006: 28 0 33
Cameron Sinclair 2006: 33 8 2
Sergey Brin + Larry Page 2007: 0 28 32
Nathalie Miebach 2011: 23 12 0
Richard Wilkinson 2011: 2 23 0
Malcolm Gladwell 2011: 0 2 1
Jay Bradner 2011: 6 14 0
Béatrice Coron 2011: 1 8 17
Hasan Elahi 2011: 34 0 28
Paul Zak 2011: 20 7 28
Anna Mracek Dietrich 2011: 0 33 2
Daniel Wolpert 2011: 7 0 34
Marco Tempest 2011: 0 32 28
Stew 2007: 5 4 1
Martin Hanczyc 2011: 0 26 25
Aparna Rao 2011: 0 27 20
Ben Kacyra 2011: 32 0 23
Allan Jones 2011: 7 31 14
Charlie Todd 2011: 15 16 1
Alexander Tsiaras 2011: 14 0 7
Yves Rossy 2011: 0 28 13
Thomas Suarez 2011: 10 16 0
Cynthia Kenyon 2011: 31 0 21
Robin Ince 2011: 4 0 28
James Howard Kunstler 2007: 8 28 15
Phil Plait 2011: 25 0 28
Péter Fankhauser 2011: 11 0 16
Joe Sabia 2011: 1 27 32
Britta Riley 2011: 29 0 8
Amy Purdy 2011: 1 0 19
Damon Horowitz 2011: 28 1 0
Annie Murphy Paul 2011: 5 29 27
John Bohannon 2011: 0 14 2
Charles Limb 2011: 12 27 0
Kathryn Schulz 2011: 0 1 28
David Kelley 2007: 33 0 18
Srdja Popovic 2011: 2 20 0
Luis von Ahn 2011: 17 32 0
Cheryl Hayashi 2011: 0 32 21
Yoav Medan 2011: 3 6 7
Stefon Harris 2011: 12 16 28
Monika Bulaj 2011: 1 20 2
Quyen Nguyen 2011: 6 0 3
Pavan Sukhdev 2011: 2 21 13
Homaro Cantu + Ben Roche 2011: 29 0 28
Ramona Pierson 2011: 1 0 28
Stewart Brand 2007: 15 2 25
Antonio Damasio 2011: 7 0 34
Sheila Nirenberg 2011: 14 7 0
Daniel Goldstein 2011: 0 20 2
Karen Tse 2011: 1 2 28
Alberto Cairo 2011: 1 28 3
AJ Jacobs 2012: 0 1 29
Jane Fonda 2012: 1 0 7
Paddy Ashdown 2012: 2 0 32
Sebastian Wernicke 2012: 17 0 20
Lauren Hodge, Shree Bose + Naomi Shah 2012: 6 3 14
Jeff Hawkins 2007: 7 28 0
Jonas Gahr Støre 2012: 2 0 28
Drew Berry 2012: 31 0 14
Morley 2012: 5 1 20
Sonaar Luthra 2012: 18 34 0
Alain de Botton 2012: 0 9 28
Mikko Hypponen 2012: 32 2 0
Clay Shirky 2012: 32 0 30
Sheena Iyengar 2012: 0 20 2
Bilal Bomani 2012: 29 18 0
Julian Baggini 2012: 0 7 18
Tierney Thys 2007: 13 0 28
Lisa Harouni 2012: 23 30 33
Diana Nyad 2012: 1 28 13
Brian Goldman 2012: 3 1 0
Gayle Tzemach Lemmon 2012: 5 2 1
Mick Mountz 2012: 0 11 8
Peter van Uhm 2012: 1 2 5
Bill Doyle 2012: 6 14 3
Shawn Achor 2012: 7 0 1
Erica Frenkel 2012: 30 3 33
Danny Hillis 2012: 0 31 14
Blaise Agüera y Arcas 2007: 0 23 32
Mike deGruy 2012: 13 21 0
Neil Burgess 2012: 14 7 0
Stephen Coleman 2012: 0 20 2
Sheikha Al Mayassa 2012: 2 5 20
Jack Horner 2012: 0 28 21
Erik Johansson 2012: 0 7 13
Drew Dudley 2012: 1 28 0
Tyrone Hayes + Penelope Jagessar Chaffer 2012: 6 0 5
Jenna McCarthy 2012: 0 28 5
Lucien Engelen 2012: 3 0 15
John Doerr 2007: 2 26 28
Simon Berrow 2012: 0 21 13
Paul Conneally 2012: 34 2 23
Garth Lenz 2012: 26 21 2
Neil MacGregor 2012: 1 2 0
Chris Bliss 2012: 0 1 17
Shilo Shiv Suleman 2012: 0 10 32
Shlomo Benartzi 2012: 20 0 2
Tan Le 2012: 1 5 17
Avi Rubin 2012: 30 0 34
Kevin Allocca 2012: 32 0 16
Ngozi Okonjo-Iweala 2007: 19 2 26
Paul Snelgrove 2012: 13 0 21
Daniel Pauly 2012: 13 21 0
Paul Gilding 2012: 2 29 25
Peter Diamandis 2012: 2 25 26
Vijay Kumar 2012: 11 0 8
Susan Cain 2012: 0 1 10
Bryan Stevenson 2012: 1 28 2
Andrew Stanton 2012: 1 0 28
James Hansen 2012: 26 25 13
Jennifer Pahlka 2012: 2 32 15
Jehane Noujaim 2006: 1 20 0
Anand Agarawala 2007: 28 0 34
Improv Everywhere 2012: 25 0 28
Larry Smith 2012: 28 0 1
Jonathan Haidt 2012: 0 31 1
Rob Reid 2012: 2 12 32
Brené Brown 2012: 28 1 5
T. Boone Pickens 2012: 26 9 2
Noel Bairey Merz 2012: 5 6 3
Taylor Wilson 2012: 26 0 10
Billy Collins 2012: 1 0 15
Peter Saul 2012: 0 3 20
Robert Thurman 2007: 24 0 1
Donald Sadoway 2012: 26 0 14
Regina Dugan 2012: 9 0 11
Leymah Gbowee 2012: 1 5 10
Ayah Bdeir 2012: 8 0 27
Marco Tempest 2012: 1 0 20
Sherry Turkle 2012: 11 0 20
Chip Kidd 2012: 1 33 0
Jack Choi 2012: 0 28 3
Lucy McRae 2012: 0 14 12
Jonathan Foley 2012: 29 18 2
David Rockwell 2007: 15 0 8
Frank Warren 2012: 1 20 28
Frans de Waal 2012: 0 21 29
Melinda Gates 2012: 5 2 10
Tal Golesworthy 2012: 3 0 30
Abigail Washburn 2012: 12 27 1
Atul Gawande 2012: 3 0 20
Drew Curtis 2012: 0 2 32
Taryn Simon 2012: 1 2 3
Laura Carstensen 2012: 20 1 0
Christina Warinner 2012: 31 3 0
Thomas Barnett 2007: 2 28 0
Brian Greene 2012: 4 26 0
Michael Norton 2012: 20 28 2
Eduardo Paes 2012: 15 0 20
Nancy Lublin 2012: 23 10 0
Joe Smith 2012: 0 20 17
Brenda Romero 2012: 16 28 1
Liz Diller 2012: 8 0 2
Amory Lovins 2012: 26 2 0
Reuben Margolin 2012: 0 28 18
Gary Kovacs 2012: 32 23 0
Stephen Lawler 2007: 23 0 25
Rory Sutherland 2012: 0 20 29
Tavi Gevinson 2012: 5 0 20
Michael Tilson Thomas 2012: 12 0 1
JP Rangaswami 2012: 34 29 1
Karen Bass 2012: 25 0 1
Joshua Foer 2012: 0 1 7
Renny Gleeson 2012: 32 0 1
Bart Knols 2012: 0 18 19
Tali Sharot 2012: 6 7 0
Jean-Baptiste Michel 2012: 17 0 27
Hans Rosling 2007: 2 19 3
David Kelley 2012: 0 30 10
Carl Schoonover 2012: 7 0 14
JR 2012: 20 1 0
Michael McDaniel 2012: 0 20 8
Nathan Wolfe 2012: 34 25 31
Hans Rosling 2012: 2 10 1
Philippe Petit 2012: 1 0 12
Shereen El-Feki 2012: 22 20 2
Reggie Watts 2012: 28 0 12
David MacKay 2012: 26 2 0
Bill Stone 2007: 0 25 8
Ken Goldberg 2012: 11 0 1
William Noel 2012: 23 0 1
Dalia Mogahed 2012: 5 2 20
Sebastian Deterding 2012: 33 0 20
Quixotic Fusion 2012: 34 8 14
Seth Shostak 2012: 28 0 10
David Birch 2012: 28 0 34
Juan Enriquez 2012: 14 0 7
Diane Kelly 2012: 0 21 3
Terry Moore 2012: 17 27 0
Dan Dennett 2007: 7 0 20
Damian Palin 2012: 18 31 13
John Hodgman 2012: 33 8 7
Ami Klin 2012: 0 7 1
John Hockenberry 2012: 33 28 1
Rebecca Onie 2012: 3 29 1
Beeban Kidron 2012: 1 10 2
Sarah Parcak 2012: 15 0 8
LZ Granderson 2012: 1 28 20
Rodney Mullen 2012: 28 0 15
Megan Kamerick 2012: 5 1 0
Alan Russell 2007: 14 3 0
David R. Dow 2012: 1 10 20
Ivan Oransky 2012: 3 0 5
Marco Tempest 2012: 1 0 15
Peter Norvig 2012: 10 0 32
Wolfgang Kessling 2012: 26 0 16
Jon Nguyen 2012: 25 0 28
Nirmalya Kumar 2012: 2 0 28
E.O. Wilson 2012: 0 17 3
Rives 2012: 25 16 28
Massimo Banzi 2012: 0 33 11
Jonathan Harris 2007: 4 0 1
Don Tapscott 2012: 32 2 0
Elyn Saks 2012: 1 3 0
Boaz Almog 2012: 26 0 4
Alanna Shaikh 2012: 0 3 28
Raghava KK 2012: 1 28 0
Cesar Harada 2012: 26 0 13
Usman Riaz + Preston Reed 2012: 9 0 28
Jane McGonigal 2012: 16 1 0
Jonathan Eisen 2012: 31 20 0
Chris Gerdes 2012: 0 7 26
Larry Brilliant 2006: 2 3 1
Emily Oster 2007: 22 19 0
Marc Goodman 2012: 31 32 0
Jared Ficklin 2012: 27 0 12
Todd Humphreys 2012: 0 1 32
Gabriel Barcia-Colombo 2012: 0 1 20
Mina Bissell 2012: 14 6 0
Jamie Drummond 2012: 2 9 0
Baba Shiv 2012: 28 6 0
Matt Mills 2012: 0 34 32
Neil Harbisson 2012: 27 0 29
John Graham-Cumming 2012: 30 28 0
Will Wright 2007: 16 0 25
Vinay Venkatraman 2012: 3 0 34
James Stavridis 2012: 2 13 5
Malte Spitz 2012: 34 32 1
Tracy Chevalier 2012: 28 1 0
Ramesh Raskar 2012: 0 4 23
Michael Hansmeyer 2012: 33 0 30
Noah Wilson-Rich 2012: 15 0 28
Michael Anti 2012: 32 2 17
Stephen Ritz 2012: 10 29 15
Daphne Koller 2012: 10 0 30
Rives 2007: 1 28 0
Becci Manson 2012: 1 18 0
Mark Applebaum 2012: 12 0 28
Scilla Elworthy 2012: 1 2 0
Margaret Heffernan 2012: 0 3 20
Max Little 2012: 3 0 27
Pam Warhurst 2012: 29 0 28
Kirby Ferguson 2012: 0 28 27
Mark Forsyth 2012: 17 2 28
Lisa Kristine 2012: 1 5 20
Ivan Krastev 2012: 2 32 20
David Bolinsky 2007: 14 0 34
Caitria + Morgan O'Neill 2012: 0 1 34
Jon Ronson 2012: 28 1 7
Timothy Prestero 2012: 33 0 3
Rob Legato 2012: 0 12 28
Robert Neuwirth 2012: 2 19 28
Shyam Sankar 2012: 30 23 0
Antony Gormley 2012: 8 0 4
Jonathan Trent 2012: 18 26 0
Kent Larson 2012: 15 0 20
Scott Fraser 2012: 1 0 28
Allison Hunt 2007: 28 1 3
Vikram Patel 2012: 3 2 20
Leslie T. Chang 2012: 1 2 15
Susan Solomon 2012: 14 22 0
Wayne McGregor 2012: 28 0 8
Beth Noveck 2012: 2 32 0
Tristram Stuart 2012: 29 2 1
Sarah-Jayne Blakemore 2012: 7 0 20
Julian Treasure 2012: 27 10 3
Andrew Blum 2012: 32 8 13
Bandi Mbubi 2012: 2 1 34
George Ayittey 2007: 19 2 20
Ed Gavagan 2012: 28 0 15
Rachel Botsman 2012: 0 20 2
Andrew McAfee 2012: 0 2 20
Read Montague 2012: 7 0 20
Clay Shirky 2012: 0 32 2
Ben Goldacre 2012: 3 22 0
Bahia Shehab 2012: 1 0 15
Aris Venetikidis 2012: 15 0 34
Vicki Arroyo 2012: 2 26 15
Amy Cuddy 2012: 0 20 28
Ngozi Okonjo-Iweala 2007: 2 19 28
Robert Gupta 2012: 12 7 3
Jason McCue 2012: 28 0 2
Shimon Schocken 2012: 30 10 8
Thomas P. Campbell 2012: 0 1 20
Tim Leberecht 2012: 0 2 32
John Maeda 2012: 33 0 30
Ruby Wax 2012: 28 7 27
Melissa Marshall 2012: 0 10 17
Maurizio Seracini 2012: 0 1 8
Eddie Obeng 2012: 0 28 18
William Kamkwamba 2007: 9 26 33
John Wilbanks 2012: 6 3 23
Beau Lotto + Amy O'Toole 2012: 16 0 28
Heather Brooke 2012: 34 2 28
Ryan Merkley 2012: 32 0 23
Pankaj Ghemawat 2012: 0 23 2
David Pizarro 2012: 0 20 5
Lemn Sissay 2012: 1 28 10
Doris Kim Sung 2012: 8 0 26
Marco Tempest 2012: 27 12 0
Rory Stewart 2012: 2 1 0
Euvin Naidoo 2007: 19 2 0
Sanjay Pradhan 2012: 2 10 1
Emma Teeling 2012: 31 0 27
Adam Garone 2012: 6 28 1
Faith Jegede 2012: 0 1 28
Matt Killingsworth 2012: 0 20 23
Jake Wood 2012: 15 1 0
Gary Greenberg 2012: 14 0 25
Georgette Mulheir 2012: 10 2 1
Jeff Hancock 2012: 0 20 28
Julie Burstein 2012: 1 0 17
Patrick Awuah 2007: 19 10 0
Arunachalam Muruganantham 2012: 30 5 2
Hannah Brencher 2012: 1 32 28
Leah Buechley 2012: 8 0 12
David Binder 2012: 15 0 1
Daphne Bavelier 2012: 16 7 0
Amos Winter 2012: 0 33 2
Sleepy Man Banjo Boys 2012: 16 1 28
Louie Schwartzberg 2012: 1 12 18
Candy Chang 2012: 1 8 20
Ernesto Sirolli 2012: 19 20 1
Nicholas Negroponte 2006: 0 10 28
Chris Abani 2007: 19 17 1
Jonas Eliasson 2012: 15 0 20
Janine Shepherd 2012: 1 28 25
Munir Virani 2012: 21 2 28
Paolo Cardini 2012: 27 0 34
Bobby Ghosh 2012: 2 1 0
Ludwick Marishane 2012: 19 0 18
Jeff Smith 2012: 0 14 28
Nina Tandon 2012: 14 6 22
Lemon Andersen 2012: 1 28 10
Ellen 't Hoen 2012: 22 2 3
Jacqueline Novogratz 2007: 19 5 2
Markham Nolan 2012: 32 0 34
Maz Jobrani 2012: 28 1 20
Marcus Byrne 2012: 0 21 19
Ben Saunders 2012: 13 1 0
Robin Chase 2012: 0 20 28
Molly Crockett 2012: 7 20 0
Steven Addis 2012: 1 0 15
Adam Davidson 2012: 2 20 0
Ronny Edry 2012: 28 1 20
Karen Thompson Walker 2013: 1 5 0
Vusi Mahlasela 2007: 27 1 28
Hadyn Parry 2013: 0 21 2
Andy Puddicombe 2013: 0 28 1
Don Levy 2013: 0 3 34
Jonathan Haidt 2013: 2 0 28
Sue Austin 2013: 0 1 20
Jarrett J. Krosoczka 2013: 1 10 28
Boghuma Kabisen Titanji 2013: 22 3 2
Angela Patton 2013: 1 5 10
Ellen Jorgensen 2013: 31 0 28
Cameron Russell 2013: 28 0 1
Jeff Skoll 2007: 1 0 20
Richard Weller 2013: 3 14 0
Colin Stokes 2013: 5 28 0
Janine di Giovanni 2013: 1 20 2
Colin Powell 2013: 10 28 1
Steven Schwaitzberg 2013: 17 0 3
Leslie Morgan Steiner 2013: 1 5 15
Wingham Rowan 2013: 2 0 32
Mitch Resnick 2013: 0 10 16
iO Tillett Wright 2013: 1 20 0
Fahad Al-Attiya 2013: 18 26 15
Dean Kamen 2007: 28 10 0
Zahra' Langhi 2013: 5 24 2
Tyler DeWitt 2013: 31 0 14
Cesar Kuriyama 2013: 0 1 28
Lee Cronin 2013: 22 0 14
Edi Rama 2013: 20 1 15
Shabana Basij-Rasikh 2013: 10 1 5
Erik Schlangen 2013: 30 0 18
James B. Glattfelder 2013: 0 2 4
Esther Perel 2013: 0 28 1
Young-ha Kim 2013: 10 1 16
Erin McKean 2007: 17 28 0
Miguel Nicolelis 2013: 7 11 16
Keith Chen 2013: 17 2 0
Afra Raymond 2013: 2 34 0
Andreas Schleicher 2013: 10 2 0
Michael Dickinson 2013: 7 0 14
Bruce Feiler 2013: 1 10 0
Bruno Maisonnier 2013: 34 8 14
Sugata Mitra 2013: 10 30 28
Jennifer Granholm 2013: 2 26 28
Amanda Palmer 2013: 12 20 1
Andrew Mwenda 2007: 19 2 32
Allan Savory 2013: 21 2 19
Edith Widder 2013: 13 0 27
Ron Finley 2013: 29 15 20
Kakenya Ntaiya 2013: 1 10 5
Shane Koyczan 2013: 1 10 28
Dan Pallotta 2013: 2 6 0
David Anderson 2013: 7 0 22
Stewart Brand 2013: 31 21 0
Bono 2013: 23 2 28
Catarina Mota 2013: 0 8 34
Theo Jansen 2007: 26 13 21
Danny Hillis 2013: 32 0 8
Elon Musk 2013: 9 26 0
Hyeonseo Lee 2013: 1 2 17
Francis Collins 2013: 14 22 0
Eric Whitacre 2013: 18 1 25
Jessica Green 2013: 8 33 0
Mark Shaw 2013: 18 0 28
Richard Turere 2013: 9 1 21
Colin Camerer 2013: 16 7 0
Kees Moeliker 2013: 1 21 0
Steven Pinker 2007: 17 0 21
Sanjay Dastoor 2013: 26 0 15
Lawrence Lessig 2013: 2 28 20
Skylar Tibbits 2013: 8 0 26
Ken Jennings 2013: 28 16 0
Freeman Hrabowski 2013: 10 0 1
Keller Rinaudo 2013: 11 0 16
Dan Ariely 2013: 20 0 1
Eric Dishman 2013: 3 0 28
Laura Snyder 2013: 0 5 1
Rose George 2013: 18 0 2
Steven Pinker 2007: 2 1 0
Thomas Insel 2013: 7 3 0
Joshua Prager 2013: 1 28 7
Andres Lozano 2013: 7 3 0
BLACK 2013: 1 0 2
John McWhorter 2013: 17 0 20
Robert Gordon 2013: 2 18 26
Erik Brynjolfsson 2013: 30 2 0
Jennifer Healey 2013: 28 11 0
David Pogue 2013: 0 32 28
Nilofer Merchant 2013: 0 6 3
Jeff Han 2006: 0 23 28
Deborah Scranton 2007: 28 1 0
Taylor Wilson 2013: 26 0 18
Sebastião Salgado 2013: 21 1 18
Juan Enriquez 2013: 28 0 23
Rita Pierson 2013: 10 28 1
Timothy Bartik 2013: 2 10 0
ShaoLan 2013: 21 17 8
Bill Gates 2013: 10 0 2
Ramsey Musallam 2013: 10 0 34
Pearl Arredondo 2013: 10 28 0
Malcolm London 2013: 8 10 1
Zeresenay Alemseged 2007: 19 21 1
Geoffrey Canada 2013: 10 28 2
Angela Lee Duckworth 2013: 10 0 8
Ken Robinson 2013: 10 20 0
Meg Jay 2013: 1 28 0
Maria Bezaitis 2013: 20 0 23
Liu Bolin 2013: 2 20 0
Jay Silver 2013: 0 28 16
Sergey Brin 2013: 0 28 34
Peter Singer 2013: 20 0 2
Phil Hansen 2013: 0 1 8
John Maeda 2007: 0 30 1
Judy MacDonald Johnston 2013: 1 3 28
Alastair Parvin 2013: 8 0 33
Ji-Hae Park 2013: 12 16 0
Paola Antonelli 2013: 33 16 0
Jackson Katz 2013: 5 0 28
Hendrik Poinar 2013: 31 0 21
Lisa Bu 2013: 1 17 10
Andrew Solomon 2013: 1 10 20
Alex Laskey 2013: 26 1 20
Anas Aremeyaw Anas 2013: 9 1 19
Stephen Petranek 2007: 25 28 0
Denise Herzing 2013: 27 18 0
Martin Villeneuve 2013: 0 25 1
Andrew McAfee 2013: 0 2 30
Raffaello D'Andrea 2013: 30 0 33
George Papandreou 2013: 2 0 33
Daniel Suarez 2013: 2 11 23
Manal al-Sharif 2013: 5 1 2
Didier Sornette 2013: 2 0 30
Juliana Rotich 2013: 32 19 0
Joseph Kim 2013: 1 29 2
Paul MacCready 2007: 0 25 26
Paul Pholeros 2013: 3 0 2
Camille Seaman 2013: 25 21 1
Lesley Hazleton 2013: 1 0 2
Peter Attia 2013: 14 3 0
Bob Mankoff 2013: 0 28 20
Michael Archer 2013: 31 21 0
Rodney Brooks 2013: 11 0 20
Eric X. Li 2013: 2 1 0
Joel Selanikio 2013: 23 0 3
Jinha Lee 2013: 34 32 0
Carolyn Porco 2007: 25 0 18
Sleepy Man Banjo Boys 2013: 16 28 1
Charmian Gooch 2013: 2 26 0
Michael Green 2013: 8 0 21
Diana Reiss, Peter Gabriel, Neil Gershenfeld and Vint Cerf 2013: 32 21 0
Jack Andraka 2013: 6 0 32
Al Vernacchio 2013: 16 0 20
Bernie Krause 2013: 27 21 0
Gavin Pretor-Pinney 2013: 0 28 20
Pico Iyer 2013: 1 0 15
Miranda Wang and Jeanny Yao 2013: 31 0 18
Kenichi Ebina 2007: 34 8 14
Tom Thum 2013: 27 0 28
John Searle 2013: 7 0 28
Kate Stone 2013: 0 26 12
Roberto D'Angelo + Francesca Fedeli 2013: 0 1 7
Paul Kemp-Robertson 2013: 0 2 20
Tania Luna 2013: 1 29 0
Bastian Schaefer 2013: 33 0 31
Eli Beer 2013: 1 3 0
Julie Taymor 2013: 1 0 28
Peter van Manen 2013: 23 3 0
Richard Branson 2007: 9 28 0
Beardyman 2013: 27 30 0
Daniel H. Cohen 2013: 0 28 17
Jinsop Lee 2013: 0 28 1
Saki Mafundikwa 2013: 19 33 10
Eleanor Longden 2013: 1 27 24
Derek Paravicini and Adam Ockelford 2013: 16 0 12
Margaret Heffernan 2013: 1 28 20
Shigeru Ban 2013: 8 20 2
Russell Foster 2013: 7 0 3
Steve Ramirez and Xu Liu 2013: 7 14 0
Hod Lipson 2007: 11 30 0
May El-Khalil 2013: 1 2 20
Adam Spencer 2013: 1 28 0
Kelly McGonigal 2013: 3 0 20
Chrystia Freeland 2013: 2 0 32
Alexa Meade 2013: 0 1 20
George Monbiot 2013: 21 0 13
Jake Barton 2013: 0 27 1
Ron McCallum 2013: 30 1 17
Sonia Shah 2013: 0 20 2
Apollo Robbins 2013: 0 16 28
Maira Kalman 2007: 28 1 0
James Lyne 2013: 32 0 23
Marla Spivak 2013: 29 21 2
Eric Berlow and Sean Gourley 2013: 0 17 28
Andras Forgacs 2013: 14 21 29
Benjamin Barber 2013: 15 2 0
Elizabeth Loftus 2013: 1 0 34
Stuart Firestein 2013: 0 28 7
Onora O'Neill 2013: 20 0 28
James Flynn 2013: 1 20 2
Kevin Breel 2013: 1 28 0
Sirena Huang 2006: 27 33 0
Jan Chipchase 2007: 33 0 20
Malcolm Gladwell 2013: 0 1 28
Kelli Swazey 2013: 1 20 3
Amy Webb 2013: 23 0 5
Fabian Oefner 2013: 0 27 18
Jason Pontin 2013: 26 2 25
Michael Porter 2013: 2 0 32
Michael Sandel 2013: 2 10 0
Janette Sadik-Khan 2013: 15 0 33
Trita Parsi 2013: 2 1 0
Gary Slutkin 2013: 3 0 2
VS Ramachandran 2007: 7 28 3
Andrew Fitzgerald 2013: 0 1 32
Jeff Speck 2013: 15 2 26
Amanda Bennett 2013: 1 6 3
Iwan Baan 2013: 8 15 0
Alessandro Acquisti 2013: 34 23 32
Hetain Patel 2013: 18 17 1
Steve Howard 2013: 26 2 0
Charles Robertson 2013: 19 2 0
Parul Sehgal 2013: 28 0 1
Gian Giudice 2013: 4 0 26
Eleni Gabre-Madhin 2007: 19 2 29
Xavier Vilalta 2013: 8 15 0
Mariana Mazzucato 2013: 0 2 32
Mohamed Hijri 2013: 29 21 2
Abha Dawesar 2013: 1 0 32
Holly Morris 2013: 5 1 0
Dong Woo Jang 2013: 1 8 0
Rodrigo Canales 2013: 2 22 0
Robin Nagle 2013: 15 1 0
Grégoire Courtine 2013: 7 11 0
Mikko Hypponen 2013: 2 32 23
Sherwin Nuland 2007: 1 0 28
Arthur Benjamin 2013: 0 10 23
Dambisa Moyo 2013: 2 20 1
Chris Downey 2013: 15 0 33
Mohamed Ali 2013: 15 1 20
Stefan Larsson 2013: 3 23 2
Jane McGonigal 2013: 16 28 0
Lian Pin Koh 2013: 21 0 8
Greg Asner 2013: 21 0 26
Henry Evans and Chad Jenkins 2013: 11 0 1
Andreas Raptopoulos 2013: 2 32 15
Matthieu Ricard 2007: 24 0 1
Peter Doolittle 2013: 0 28 17
Jared Diamond 2013: 20 1 3
Suzana Herculano-Houzel 2013: 7 26 0
David Steindl-Rast 2013: 0 1 20
Toby Eccles 2013: 2 0 20
Geraldine Hamilton 2013: 14 0 22
Sally Kohn 2013: 24 28 0
David Lang 2013: 11 13 33
Enrique Peñalosa 2013: 15 2 0
Boyd Varty 2013: 19 1 18
Lawrence Lessig 2007: 12 32 0
Diébédo Francis Kéré 2013: 8 10 1
Eddy Cartaya 2013: 25 0 18
Stephen Cave 2013: 1 0 20
Rose George 2013: 13 0 28
Toni Griffin 2013: 15 0 2
Marco Annunziata 2013: 30 2 23
Andrew Solomon 2013: 1 20 28
Krista Donaldson 2013: 0 2 33
Paul Piff 2013: 16 2 20
Diana Nyad 2013: 1 13 28
Paul Rothemund 2007: 31 30 0
Mick Cornett 2014: 15 0 1
Maysoon Zayid 2014: 1 16 32
Suzanne Talhouk 2014: 17 1 28
Roger Stein 2014: 6 22 0
Sandra Aamodt 2014: 29 7 0
Frederic Kaplan 2014: 34 0 8
Ryan Holladay 2014: 12 0 15
Harish Manwani 2014: 2 0 18
Mark Kendall 2014: 14 0 2
Sheryl Sandberg 2014: 5 1 28
David Keith 2007: 0 25 28
Luke Syson 2014: 0 1 4
Guy Hoffman 2014: 11 0 12
Shereen El Feki 2014: 5 22 1
Paula Johnson 2014: 5 6 3
Yves Morieux 2014: 0 2 33
Joe Kowan 2014: 27 28 0
Anant Agarwal 2014: 10 0 32
Anne Milgram 2014: 23 2 0
McKenna Pope 2014: 0 32 1
Nicolas Perony 2014: 21 0 11
Juan Enriquez 2007: 26 0 28
Maya Penn 2014: 1 0 2
Esta Soler 2014: 5 1 23
Dan Berkenstock 2014: 23 0 33
Teddy Cruz 2014: 15 2 8
Alex Wissner-Gross 2014: 30 16 25
Aparna Rao 2014: 0 27 7
David Puttnam 2014: 2 3 1
Leyla Acaroglu 2014: 0 33 29
Chris McKnett 2014: 2 18 0
Rupal Patel 2014: 27 1 0
Larry Brilliant 2007: 2 21 1
Yann Dall'Aglio 2014: 1 0 2
Molly Stevens 2014: 14 0 3
Roselinde Torres 2014: 2 0 19
Christopher Ryan 2014: 5 0 21
Ash Beckham 2014: 1 28 0
Siddharthan Chandran 2014: 14 7 3
Catherine Bracy 2014: 15 2 0
Michael Metcalfe 2014: 2 0 19
Henry Lin 2014: 4 0 26
Siddharthan Chandran 2014: 14 7 3
Jennifer Lin 2006: 28 0 12
Robert Full 2007: 11 0 33
Catherine Bracy 2014: 15 2 0
Henry Lin 2014: 4 0 26
Annette Heuser 2014: 2 0 19
Mary Lou Jepsen 2014: 7 0 32
Philip Evans 2014: 23 0 2
Christopher Soghoian 2014: 2 32 0
Gabe Barcia-Colombo 2014: 31 30 0
Manu Prakash 2014: 0 3 10
Ajit Narayanan 2014: 17 0 10
Anne-Marie Slaughter 2014: 5 2 0
Ron Eglash 2007: 19 0 8
Toby Shapshak 2014: 19 0 2
Carin Bondar 2014: 21 0 5
Steven Pinker and Rebecca Newberger Goldstein 2014: 21 2 1
Daniel Reisel 2014: 7 14 0
Edward Snowden 2014: 9 32 2
Chris Hadfield 2014: 0 4 25
Charmian Gooch 2014: 2 0 32
Richard Ledgett 2014: 9 0 2
Larry Page 2014: 0 32 28
Ziauddin Yousafzai 2014: 1 5 10
Philippe Starck 2007: 33 28 1
Bran Ferren 2014: 33 0 1
Ed Yong 2014: 7 0 21
Del Harvey 2014: 0 34 20
Hugh Herr 2014: 33 14 0
Geena Rocero 2014: 1 5 0
TED staff 2014: 28 12 0
Allan Adams 2014: 4 0 28
Bill and Melinda Gates 2014: 9 0 10
Jennifer Golbeck 2014: 23 32 20
Lawrence Lessig 2014: 2 1 28
Murray Gell-Mann 2007: 4 0 28
Amanda Burden 2014: 15 8 0
Christopher Emdin 2014: 10 0 28
Louie Schwartzberg 2014: 31 0 23
David Sengeh 2014: 3 1 5
Gabby Giffords and Mark Kelly 2014: 1 0 28
David Brooks 2014: 0 1 8
Jennifer Senior 2014: 10 1 0
Norman Spack 2014: 3 1 10
Jeremy Kasdin 2014: 25 4 0
Matthew Carter 2014: 33 0 23
Amory Lovins 2007: 26 2 0
Sarah Lewis 2014: 1 0 5
Michel Laberge 2014: 26 30 0
Hamish Jolly 2014: 18 0 13
James Patten 2014: 16 0 11
Elizabeth Gilbert 2014: 1 28 0
Wendy Chung 2014: 31 0 7
David Epstein 2014: 0 1 18
Andrew Bastawrous 2014: 3 20 0
Gavin Schmidt 2014: 0 25 13
Sarah Jones 2014: 28 0 5
Arthur Benjamin 2007: 28 9 0
Stanley McChrystal 2014: 34 28 20
Randall Munroe 2014: 23 0 28
Mark Ronson 2014: 12 27 0
William Black 2014: 2 0 28
Deborah Gordon 2014: 6 0 21
Kevin Briggs 2014: 1 0 28
Tristram Wyatt 2014: 0 21 7
Rives 2014: 1 0 28
Simon Sinek 2014: 20 1 2
Jackie Savitz 2014: 13 2 29
Daniel Goleman 2007: 24 0 28
Andrew Solomon 2014: 1 28 2
Chris Kluwe 2014: 16 0 34
Wes Moore 2014: 1 0 20
Sebastian Junger 2014: 1 5 0
Jon Mooallem 2014: 21 0 1
Kitra Cahana 2014: 1 29 2
Stephen Friend 2014: 0 3 22
Sting 2014: 1 28 13
Ray Kurzweil 2014: 7 0 17
Dan Gilbert 2014: 20 0 1
Lakshmi Pratury 2007: 1 6 0
Stephen Burt 2014: 17 1 28
Robert Full 2014: 11 0 21
Yoruba Richen 2014: 1 2 19
Stella Young 2014: 20 28 0
Keren Elazari 2014: 32 2 0
Will Potter 2014: 1 21 2
Uri Alon 2014: 0 28 1
AJ Jacobs 2014: 28 31 21
Kwame Anthony Appiah 2014: 0 1 28
Anne Curzan 2014: 17 0 28
Gever Tulley 2007: 10 0 16
Ruth Chang 2014: 0 2 1
Jamila Lyiscott 2014: 17 28 1
Billy Collins 2014: 1 29 0
Shaka Senghor 2014: 1 5 0
Lorrie Faith Cranor 2014: 0 20 23
Naomi Oreskes 2014: 0 25 4
Ge Wang 2014: 12 27 0
Julian Treasure 2014: 27 0 28
Chris Domas 2014: 34 0 30
Sara Lewis 2014: 0 21 1
Isabel Allende 2008: 5 1 2
Simon Anholt 2014: 2 0 28
Paul Bloom 2014: 20 1 0
George Takei 2014: 1 2 5
Joi Ito 2014: 32 0 2
Nicholas Negroponte 2014: 0 32 9
Renata Salecl 2014: 0 20 1
Karima Bennoune 2014: 1 5 2
David Kwong 2014: 28 0 7
David Chalmers 2014: 7 0 4
Nikolai Begg 2014: 0 3 12
Amy Smith 2006: 0 18 2
Yossi Vardi 2008: 32 0 3
Shih Chieh Huang 2014: 0 18 4
Heather Barnett 2014: 0 29 14
Ze Frank 2014: 1 0 34
Shai Reshef 2014: 10 2 0
Margaret Gould Stewart 2014: 33 0 20
Hubertus Knabe 2014: 20 2 1
Janet Iwasa 2014: 22 14 0
Megan Washington 2014: 28 17 0
Talithia Williams 2014: 23 3 28
Nick Hanauer 2014: 2 0 15
Deborah Gordon 2008: 0 29 1
Dan Pacholke 2014: 14 0 1
Eric Liu 2014: 15 2 0
Clint Smith 2014: 1 10 27
Tim Berners-Lee 2014: 32 23 0
Aziza Chaouni 2014: 15 18 8
Jarrett J. Krosoczka 2014: 10 1 28
Laurel Braitman 2014: 21 0 1
Ziyah Gafić 2014: 1 0 2
Martin Rees 2014: 25 2 4
Rose Goslinga 2014: 2 19 0
J.J. Abrams 2008: 28 0 1
Meera Vijayann 2014: 5 1 2
Sally Kohn 2014: 32 28 5
Jill Shargaa 2014: 29 1 32
Jim Holt 2014: 4 28 0
Isabel Allende 2014: 1 7 28
Shubhendu Sharma 2014: 21 0 26
Colin Grant 2014: 1 27 10
Zak Ebrahim 2014: 1 20 0
Dan Barasch 2014: 15 8 0
Hans and Ola Rosling 2014: 2 0 10
David Gallo 2008: 13 0 18
Uldus Bakhtiozina 2014: 1 5 0
Rishi Manchanda 2014: 3 0 18
Andrew Connolly 2014: 4 23 25
Mac Barnett 2014: 10 28 1
Avi Reichental 2014: 0 28 1
Antonio Donato Nobre 2014: 21 18 13
Lord Nicholas Stern 2014: 15 2 26
Kenneth Cukier 2014: 23 30 34
Eman Mohammed 2014: 5 1 8
Matthew O'Reilly 2014: 3 1 0
Paola Antonelli 2008: 33 0 1
Moshe Safdie 2014: 8 15 2
Francis de los Reyes 2014: 18 0 2
Susan Colantuono 2014: 5 2 0
Gail Reed 2014: 3 2 10
Nancy Kanwisher 2014: 7 0 28
Daria van den Bercken 2014: 12 1 27
Thomas Piketty 2014: 2 23 0
Meaghan Ramsey 2014: 0 10 5
Pia Mancini 2014: 2 0 34
Dilip Ratha 2014: 2 20 19
Frank Gehry 2008: 8 28 0
Glenn Greenwald 2014: 20 0 32
Jeff Iliff 2014: 7 14 0
Myriam Sidibe 2014: 2 3 10
Jorge Soto 2014: 6 31 0
Melissa Fleming 2014: 1 10 2
Kitra Cahana 2014: 1 27 30
Susan Etlinger 2014: 23 0 20
Fred Swaniker 2014: 19 2 0
Joy Sun 2014: 2 20 0
Fabien Cousteau 2014: 13 0 21
Raul Midon 2008: 28 27 12
Marc Abrahams 2014: 3 20 0
Kimberley Motley 2014: 2 1 34
Sergei Lupashin 2014: 0 20 28
Frans Lanting 2014: 21 1 0
Debra Jarvis 2014: 6 1 28
Jeremy Heimans 2014: 0 2 26
Alessandra Orofino 2014: 15 2 0
Ameenah Gurib-Fakim 2014: 21 29 19
Kare Anderson 2014: 20 0 1
Alejandro Aravena 2014: 33 8 15
Bill Strickland 2008: 10 1 8
Haas&Hahn 2014: 1 0 15
Ramanan Laxminarayan 2014: 26 3 0
Michael Green 2014: 2 0 23
Ethan Nadelmann 2014: 22 2 3
Leana Wen 2014: 3 1 28
Vincent Moon and Naná Vasconcelos 2014: 12 0 32
David Grady 2014: 0 34 20
Will Marshall 2014: 25 23 0
Nancy Frates 2014: 1 28 3
Joe Landolina 2014: 14 0 21
Ben Dunlap 2008: 1 28 12
Rosie King 2014: 20 0 28
Mark Plotkin 2014: 21 1 2
Emily Balcetis 2014: 20 0 1
Pico Iyer 2014: 1 0 12
Oren Yakobovich 2014: 1 0 2
Ben Saunders 2014: 1 29 0
Rainer Strack 2014: 2 0 11
Barbara Natterson-Horowitz 2014: 3 21 6
Aakash Odedra 2014: 34 8 14
Jose Miguel Sokoloff 2014: 1 28 21
David Pogue 2008: 12 28 32
Anastasia Taylor-Lind 2014: 5 1 0
Thomas Hellum 2014: 1 0 20
Catherine Crump 2014: 34 23 2
Dave Troy 2014: 15 12 23
Vernā Myers 2014: 28 5 1
Jeremy Howard 2014: 30 0 23
Carol Dweck 2014: 10 0 16
Bruno Torturra 2014: 32 0 20
Mundano 2014: 15 2 4
Erin McKean 2014: 17 0 20
Ross Lovegrove 2006: 0 33 18
Alison Jackson 2008: 0 1 34
Michael Rubinstein 2014: 27 0 16
Asha de Vos 2015: 13 21 0
Daniele Quercia 2015: 15 0 1
Aziz Abu Sarah 2015: 29 1 0
Fredy Peccerelli 2015: 31 1 5
Tasso Azevedo 2015: 21 2 26
Navi Radjou 2015: 2 26 18
Robert Swan 2015: 25 13 1
Robert Muggah 2015: 15 2 0
Cristina Domenech 2015: 17 1 0
Chris Anderson 2008: 0 7 28
Matthieu Ricard 2015: 2 1 0
Sarah Bergbreiter 2015: 11 0 26
Joe Madiath 2015: 18 2 5
Morgana Bailey 2015: 1 0 2
Miguel Nicolelis 2015: 7 0 30
Severine Autesserre 2015: 2 1 5
Khadija Gbla 2015: 1 28 5
Bassam Tariq 2015: 1 15 0
Zeynep Tufekci 2015: 32 2 0
Bruce Aylward 2015: 3 2 19
Robin Chase 2008: 0 15 2
Ben Ambridge 2015: 5 7 0
Tom Wujec 2015: 0 33 20
Brian Dettmer 2015: 0 34 17
Jaap de Roode 2015: 21 22 3
Ricardo Semler 2015: 28 10 20
Kenneth Shinozuka 2015: 3 33 0
Hannah Fry 2015: 0 20 28
Guy Winch 2015: 1 3 0
Nadine Burke Harris 2015: 3 7 0
Laura Boushnak 2015: 5 10 2
Jaime Lerner 2008: 15 33 0
Angelo Vermeulen 2015: 8 0 25
James A. White Sr. 2015: 28 1 20
Rob Knight 2015: 31 0 3
Khalida Brohi 2015: 9 5 1
Romina Libster 2015: 3 20 15
Ben Wellington 2015: 23 15 0
Helder Guimarães 2015: 0 28 1
Jon Gosier 2015: 0 2 32
Topher White 2015: 21 27 0
Harry Baker 2015: 1 20 0
David Macaulay 2008: 0 15 8
Andy Yen 2015: 32 0 23
Ilona Szabó de Carvalho 2015: 22 2 1
Sangu Delle 2015: 19 2 29
Marc Kushner 2015: 8 0 2
Ismael Nazario 2015: 28 1 20
Shimpei Takahashi 2015: 23 16 0
Linda Hill 2015: 0 20 2
Vincent Cochetel 2015: 1 2 28
Robyn Stein DeLuca 2015: 5 3 0
David Eagleman 2015: 7 0 23
Michael Pollan 2008: 21 0 29
Joseph DeSimone 2015: 0 33 16
Monica Lewinsky 2015: 24 32 1
Fei-Fei Li 2015: 30 0 23
Anand Giridharadas 2015: 1 2 5
Dave Isay 2015: 1 0 20
Theaster Gates 2015: 8 0 15
Dame Stephanie Shirley 2015: 5 1 0
Alison Killing 2015: 8 3 1
Daniel Kish 2015: 27 0 9
Kevin Rudd 2015: 2 1 0
Howard Rheingold 2008: 16 0 2
Boniface Mwangi 2015: 1 2 20
Bill Gates 2015: 2 0 3
Bel Pesce 2015: 0 20 1
Eduardo Sáenz de Cabezón 2015: 0 28 10
Dan Ariely 2015: 20 0 2
Fred Jansen 2015: 25 0 26
Barat Ali Batoor 2015: 1 18 20
Kailash Satyarthi 2015: 1 10 2
Takaharu Tezuka 2015: 10 8 28
Paul Tudor Jones II 2015: 2 0 28
Pamelia Kurstin 2008: 27 28 16
Nathalie Cabrol 2015: 25 0 30
Gary Haugen 2015: 24 2 1
Jedidah Isler 2015: 4 23 25
Chris Milk 2015: 1 30 0
Clint Smith 2015: 1 0 10
Nizar Ibrahim 2015: 18 21 0
Nick Bostrom 2015: 30 0 21
Greg Gage 2015: 7 28 0
Sophie Scott 2015: 27 0 28
Alice Goffman 2015: 10 1 20
George Dyson 2008: 8 28 0
Pamela Ronald 2015: 31 29 21
Abe Davis 2015: 27 0 16
Bill T. Jones 2015: 28 25 9
Tal Danino 2015: 31 6 4
Dawn Landes 2015: 13 1 0
Anand Varma 2015: 14 0 29
Elora Hardy 2015: 8 0 1
Roman Mars 2015: 33 15 28
The Lady Lifers 2015: 1 5 2
Martine Rothblatt 2015: 9 1 0
Moshe Safdie 2008: 8 1 0
Cosmin Mihaiu 2015: 3 16 0
Steven Wise 2015: 0 1 21
Esther Perel 2015: 1 0 5
Chris Burkard 2015: 18 0 1
Jeffrey Brown 2015: 1 15 2
Yassmin Abdel-Magied 2015: 0 28 5
Sara Seager 2015: 25 0 4
Jimmy Nelson 2015: 28 1 0
Bill Gross 2015: 0 2 32
Laura Schulz 2015: 0 23 7
Richard Baraniuk 2006: 0 12 10
Jill Sobule + Julia Sweeney 2008: 28 1 4
Tony Fadell 2015: 0 1 26
Trevor Aaronson 2015: 2 1 0
Linda Cliatt-Wayman 2015: 10 1 0
Suki Kim 2015: 1 10 0
Sarah Jones 2015: 28 5 1
Donald Hoffman 2015: 0 7 4
Lee Mokobe 2015: 1 10 16
Rana el Kaliouby 2015: 0 5 30
Margaret Heffernan 2015: 0 20 2
Steve Silberman 2015: 10 1 3
Raspyni Brothers 2008: 28 0 33
LaToya Ruby Frazier 2015: 1 6 5
Joey Alexander 2015: 34 8 14
Roxane Gay 2015: 5 12 0
Chip Kidd 2015: 0 28 33
Maryn McKenna 2015: 3 1 22
Chris Urmson 2015: 0 28 23
Dame Ellen MacArthur 2015: 1 13 26
Jimmy Carter 2015: 5 2 1
Latif Nasser 2015: 3 1 0
Gayle Tzemach Lemmon 2015: 5 1 0
Joseph Lekuton 2008: 1 10 18
Rajiv Maheswaran 2015: 30 16 0
Memory Banda 2015: 5 1 28
Ash Beckham 2015: 1 0 28
Noy Thrupkaew 2015: 1 2 0
Aspen Baker 2015: 1 27 5
Alec Soth + Stacey Baker 2015: 1 5 0
Salvatore Iaconesi 2015: 6 3 7
Marlene Zuk 2015: 21 0 20
Jon Ronson 2015: 1 20 28
Alaa Murabit 2015: 5 1 10
Steve Jurvetson 2008: 0 10 30
John Green 2015: 0 1 32
eL Seed 2015: 1 17 20
Yuval Noah Harari 2015: 21 2 0
Benedetta Berti 2015: 2 32 0
Rich Benjamin 2015: 1 20 2
Matt Kenyon 2015: 1 32 0
Patience Mthunzi 2015: 22 14 3
Alix Generous 2015: 1 0 20
Manuel Lima 2015: 21 0 7
Tony Wyss-Coray 2015: 7 14 0
Roy Gould + Curtis Wong 2008: 4 0 28
Christopher Soghoian 2015: 2 32 0
Dustin Yellin 2015: 13 0 21
Jim Al-Khalili 2015: 31 0 4
Seth Berkley 2015: 2 3 0
Robin Murphy 2015: 11 23 0
Yves Morieux 2015: 26 0 20
Wendy Freedman 2015: 4 25 0
Elizabeth Nyamayaro 2015: 5 1 2
Jamie Bartlett 2015: 32 0 22
Jim Simons 2015: 9 0 1
Alan Kay 2008: 0 10 7
Alan Eustace 2015: 0 25 28
Barry Schwartz 2015: 20 0 21
BJ Miller 2015: 3 0 1
Billie Jean King 2015: 5 28 0
David Rothkopf 2015: 2 32 28
Mia Birdsong 2015: 20 10 1
Michael Kimmel 2015: 5 1 28
Mandy Len Catron 2015: 1 0 28
Scott Dinsmore 2015: 20 0 1
Sakena Yacoobi 2015: 1 5 10
Craig Venter 2008: 31 0 33
Frances Larson 2015: 32 20 0
Mary Robinson 2015: 2 26 1
Robin Morgan 2015: 7 0 1
Samuel Cohen 2015: 3 6 7
Taiye Selasi 2015: 1 2 17
Mac Stone 2015: 18 0 21
Martin Pistorius 2015: 1 17 0
Emilie Wapnick 2015: 0 10 1
Alice Bows-Larkin 2015: 26 2 0
Siddhartha Mukherjee 2015: 14 6 0
Nicholas Negroponte 2008: 0 30 10
Neri Oxman 2015: 33 8 0
Teitur 2015: 1 28 27
Vijay Kumar 2015: 11 0 8
Alyson McGregor 2015: 5 3 0
Anders Fjellberg 2015: 1 31 17
Meklit Hadero 2015: 12 17 27
Will Potter 2015: 1 2 0
Jennifer Doudna 2015: 31 14 0
Tom Uglow 2015: 34 32 0
Francesco Sauro 2015: 25 0 13
Jill Bolte Taylor 2008: 7 1 26
Hilary Cottam 2015: 2 0 20
Cesar Harada 2015: 18 10 11
Christine Sun Kim 2015: 27 12 0
Mathias Jud 2015: 34 2 32
Daniel Levitin 2015: 3 7 0
Nancy Lublin 2015: 23 1 2
Melissa Fleming 2015: 1 18 20
Patrícia Medici 2015: 21 0 23
Harald Haas 2015: 26 14 25
Kaki King 2015: 34 8 14
Frank Gehry 2008: 8 0 13
Jenni Chang and Lisa Dazols 2015: 1 2 20
Andreas Ekström 2015: 32 0 10
Chelsea Shields 2015: 5 1 2
Jean-Paul Mari 2015: 1 20 17
Josh Luber 2015: 2 23 0
Nonny de la Peña 2015: 0 1 29
Anote Tong 2015: 9 13 2
Carl Safina 2015: 21 7 0
Genevieve von Petzinger 2015: 0 21 17
Ann Morgan 2015: 17 1 0
Majora Carter 2006: 15 2 1
Jimmy Wales 2006: 32 0 20
Dave Eggers 2008: 10 28 0
Regina Hartley 2015: 1 10 2
Marina Abramović 2015: 0 1 28
Kristen Marhaver 2015: 21 0 13
Jessica Shortall 2015: 5 1 0
Chieko Asakawa 2015: 32 9 0
Guillaume Néry 2015: 0 18 25
Jedidah Isler 2015: 5 4 1
Danit Peleg 2015: 0 33 12
Raymond Wang 2015: 0 3 30
Nicole Paris and Ed Cage 2015: 28 0 1
Karen Armstrong 2008: 24 1 20
Paul Greenberg 2015: 13 0 29
Lucianne Walkowicz 2015: 25 4 0
Alison Killing 2015: 20 0 15
Jane Fonda and Lily Tomlin 2015: 5 28 0
António Guterres 2015: 2 20 0
Rodrigo Bijou 2015: 32 2 34
Jason deCaires Taylor 2015: 13 0 8
Robert Waldinger 2015: 1 20 3
Harry Cliff 2016: 4 0 26
Sebastian Wernicke 2016: 23 0 7
Neil Turok 2008: 19 4 10
Aomawa Shields 2016: 25 4 18
David Sedlak 2016: 18 15 0
James Veitch 2016: 28 1 3
Tim Harford 2016: 0 10 1
Melvin Russell 2016: 1 15 28
Wael Ghonim 2016: 32 20 1
Ole Scheeren 2016: 8 0 1
Jill Farrant 2016: 21 29 18
Oscar Schwartz 2016: 30 17 0
Achenyo Idachaba 2016: 18 1 29
Norman Foster 2008: 8 26 0
Elizabeth Lev 2016: 1 0 8
Yanis Varoufakis 2016: 2 0 5
David Gruber 2016: 13 0 7
Tania Simoncelli 2016: 31 6 0
Auke Ijspeert 2016: 11 0 21
Melati and Isabel Wijsen 2016: 10 1 2
Linda Liukas 2016: 30 0 10
Andrés Ruzo 2016: 18 21 1
Judson Brewer 2016: 29 7 0
Pardis Sabeti 2016: 3 23 0
Christopher deCharms 2008: 7 0 3
Matthew Williams 2016: 16 20 1
Dambisa Moyo 2016: 2 0 8
Sean Follmer 2016: 0 34 32
Gregory Heyworth 2016: 19 0 1
Mike Velings 2016: 13 29 2
Dorothy Roberts 2016: 3 0 20
Jocelyne Bloch 2016: 14 7 0
Celeste Headlee 2016: 28 0 20
Shonda Rhimes 2016: 16 1 28
Allan Adams 2016: 4 27 0
Clifford Stoll 2008: 28 27 4
Raffaello D'Andrea 2016: 30 0 4
Al Gore 2016: 26 2 0
Dalia Mogahed 2016: 1 20 0
Audrey Choi 2016: 2 0 13
Mary Bassett 2016: 3 2 22
Ivan Coyote 2016: 1 0 28
Thomas Peschak 2016: 13 0 1
Magda Sayeg 2016: 0 1 15
Russ Altman 2016: 22 3 23
Alexander Betts 2016: 2 0 15
Siegfried Woldhek 2008: 1 0 21
Travis Kalanick 2016: 15 9 0
Reshma Saujani 2016: 5 9 10
Caleb Harper 2016: 29 21 0
Laura Robinson 2016: 13 0 28
Mileha Soneji 2016: 0 1 3
Tshering Tobgay 2016: 2 26 21
Joe Gebbia 2016: 33 0 1
Tim Urban 2016: 0 7 1
Jessica Ladd 2016: 34 0 1
Arthur Brooks 2016: 2 28 1
David Hoffman 2008: 1 32 0
Meron Gribetz 2016: 7 30 0
Adam Foss 2016: 1 10 2
Carol Fishman Cohen 2016: 1 0 20
Latif Nasser 2016: 0 21 1
Siyanda Mohutsiwa 2016: 19 2 1
Alex Kipman 2016: 30 4 0
Angélica Dass 2016: 1 10 0
Dan Gross 2016: 2 1 0
Lisa Nip 2016: 25 21 31
Knut Haanaes 2016: 0 2 28
Jakob Trollback 2008: 12 1 28
Adam Grant 2016: 0 32 20
Haley Van Dyck 2016: 2 32 0
Parag Khanna 2016: 2 15 19
Danielle Feinberg 2016: 11 0 1
Tabetha Boyajian 2016: 25 23 0
Robert Palmer 2016: 2 0 34
Linus Torvalds 2016: 9 0 20
Hugh Evans 2016: 2 1 10
Stephen Petranek 2016: 25 18 0
Paula Hammond 2016: 6 14 31
Stephen Hawking 2008: 4 25 2
Astro Teller 2016: 0 26 32
Mary Norris 2016: 0 1 12
Christiana Figueres 2016: 2 26 0
Joshua Prager 2016: 1 0 25
Chris Anderson 2016: 0 9 7
Juan Enriquez 2016: 21 0 20
Aditi Gupta 2016: 5 10 1
Kenneth Lacovara 2016: 25 21 0
Shivani Siroya 2016: 23 2 29
R. Luke DuBois 2016: 0 23 15
Mena Trott 2006: 1 20 28
Al Gore 2008: 2 26 25
Ameera Harouda 2016: 1 27 5
Michael Metcalfe 2016: 2 26 0
Riccardo Sabatini 2016: 30 0 34
Sarah Gray 2016: 1 3 0
Alice Rawsthorn 2016: 33 3 32
Dan Pallotta 2016: 0 1 24
Monica Byrne 2016: 4 1 28
Jennifer Kahn 2016: 31 0 21
Uri Hasson 2016: 7 17 27
Sanford Biggers 2016: 1 0 2
Johnny Lee 2008: 16 0 30
Sangeeta Bhatia 2016: 6 0 14
Kang Lee 2016: 10 1 16
Moran Cerf 2016: 7 0 14
Tod Machover + Dan Ellsey 2008: 12 0 20
Yochai Benkler 2008: 0 34 32
Ernest Madu 2008: 3 19 2
Amy Tan 2008: 1 4 0
Brian Greene 2008: 4 0 8
Brian Cox 2008: 4 0 18
They Might Be Giants 2008: 28 27 1
Hector Ruiz 2008: 0 10 19
Ze Frank 2006: 28 0 32
Paul Stamets 2008: 21 25 26
Paul Ewald 2008: 18 0 3
Michael Moschen 2008: 0 12 28
Joshua Klein 2008: 30 0 21
Alisa Miller 2008: 2 32 1
Mark Bittman 2008: 29 21 0
Robert Ballard 2008: 13 25 0
Yves Behar 2008: 33 0 10
Arthur Ganson 2008: 30 0 28
Seyi Oyesola 2008: 28 3 19
Helen Fisher 2006: 5 7 1
Paul Collier 2008: 2 24 0
Susan Blackmore 2008: 0 30 34
Nathan Myhrvold 2008: 0 28 29
Wade Davis 2008: 1 25 13
Murray Gell-Mann 2008: 17 9 0
George Dyson 2008: 30 4 0
Chris Jordan 2008: 0 2 5
Dean Ornish 2008: 31 7 6
Robert Full 2008: 11 21 0
Adam Grosser 2008: 0 18 20
Eve Ensler 2006: 5 1 0
Steven Levitt 2008: 23 0 10
Benjamin Zander 2008: 12 28 1
Nicholas Negroponte 2008: 28 10 0
Nellie McKay 2008: 31 28 6
Peter Diamandis 2008: 1 25 28
Rick Smolan 2008: 1 28 10
Raul Midon 2008: 25 28 2
Corneille Ewango 2008: 21 1 28
Torsten Reil 2008: 0 16 30
David Hoffman 2008: 1 28 0
David Deutsch 2006: 4 25 0
Clay Shirky 2008: 0 2 20
Nellie McKay 2008: 28 13 1
Freeman Dyson 2008: 25 13 4
Helen Fisher 2008: 7 1 28
Billy Graham 2008: 1 28 24
AJ Jacobs 2008: 1 0 28
Keith Barry 2008: 28 0 9
Martin Seligman 2008: 33 0 1
Marisa Fick-Jordan 2008: 0 33 8
Chris Abani 2008: 1 5 28
Richard Dawkins 2006: 4 0 25
Louise Leakey 2008: 21 25 19
Jonathan Harris 2008: 1 0 13
Reed Kroloff 2008: 8 0 30
Kevin Kelly 2008: 30 32 0
Kwabena Boahen 2008: 7 23 0
Robert Lang 2008: 0 33 8
Bruno Bowden + Rufus Cappadocia 2008: 28 0 12
Patricia Burchat 2008: 4 26 0
Spencer Wells 2008: 31 19 0
David Griffin 2008: 1 0 13
Malcolm Gladwell 2006: 0 23 29
Lennart Green 2008: 28 9 7
Ian Dunbar 2008: 28 0 1
Nellie McKay 2008: 28 1 3
John Q. Walker 2008: 28 16 0
Sugata Mitra 2008: 10 17 30
Ory Okolloh 2008: 28 19 10
Einstein the Parrot 2008: 28 0 18
Paul Rothemund 2008: 31 0 30
Peter Diamandis 2008: 25 2 28
Peter Hirshberg 2008: 32 30 0
Steven Levitt 2006: 0 28 1
Jonathan Drori 2008: 0 28 10
Jane Goodall 2008: 19 21 0
Irwin Redlener 2008: 26 0 2
Brewster Kahle 2008: 0 32 28
David Gallo 2008: 13 21 0
Carmen Agra Deedy 2008: 28 1 5
Keith Bellows 2008: 28 0 33
Ann Cooper 2008: 29 10 28
Jonathan Haidt 2008: 20 0 28
Eve Ensler 2008: 5 1 0
Barry Schwartz 2006: 0 20 28
David S. Rose 2008: 28 0 2
Marvin Minsky 2008: 0 20 28
Philip Zimbardo 2008: 28 20 1
Laura Trice 2008: 28 1 23
Caleb Chung 2008: 28 0 11
Steven Pinker 2008: 0 1 10
Rodney Brooks 2008: 11 28 0
Stefan Sagmeister 2008: 33 0 1
Noah Feldman 2008: 2 20 0
Liz Diller 2008: 8 0 18
Ken Robinson 2006: 10 1 28
Dan Gilbert 2006: 0 28 1
James Nachtwey 2008: 3 2 1
David Perry 2008: 16 0 10
Doris Kearns Goodwin 2008: 1 16 28
Steven Johnson 2008: 0 32 15
James Burchfield 2008: 28 0 8
Garrett Lisi 2008: 4 0 9
Paola Antonelli 2008: 33 0 30
Virginia Postrel 2008: 0 1 28
Dean Ornish 2008: 3 29 6
John Hodgman 2008: 1 4 28
Eva Vertes 2006: 6 14 0
Paul MacCready 2008: 0 25 26
Mihaly Csikszentmihalyi 2008: 0 20 1
Kristen Ashburn 2008: 22 9 1
Jared Diamond 2008: 2 0 21
Rives 2008: 28 12 13
Zach Kaplan + Keith Schacht 2008: 0 18 28
Newton Aduaka 2008: 1 28 0
Jackie Tabick 2008: 24 1 28
Dayananda Saraswati 2008: 24 0 17
James Forbes 2008: 24 1 28
Aubrey de Grey 2006: 0 14 28
Feisal Abdul Rauf 2008: 24 1 28
Robert Thurman 2008: 24 1 0
Robert Wright 2008: 24 0 20
Graham Hawkes 2008: 13 25 28
James Surowiecki 2008: 32 0 20
John Francis 2008: 28 1 26
Tim Brown 2008: 16 0 28
Luca Turin 2008: 0 28 4
Lee Smolin 2008: 4 0 2
Samantha Power 2008: 2 0 1
Iqbal Quadir 2006: 2 20 0
Charles Elachi 2008: 25 0 28
Ursus Wehrli 2008: 0 28 17
Stewart Brand 2008: 0 21 8
Isaac Mizrahi 2008: 28 0 33
Franco Sacchi 2008: 19 9 1
George Smoot 2008: 4 0 25
Bill Joy 2008: 0 2 26
Dan Barber 2008: 29 28 0
Andy Hobsbawm 2008: 0 1 20
Gregory Petsko 2008: 3 6 7
Jacqueline Novogratz 2006: 2 19 0
Richard Preston 2008: 21 0 1
Philip Rosedale 2008: 0 32 28
Larry Burns 2008: 2 0 14
Nick Sears 2008: 0 15 23
David Holt 2008: 28 1 12
Eva Zeisel 2008: 1 0 33
Dennis vanEngelsdorp 2008: 0 21 28
Jay Walker 2008: 7 0 30
Dan Gilbert 2008: 20 0 9
Benjamin Wallace 2008: 28 0 26
Ashraf Ghani 2006: 2 0 9
Penelope Boston 2008: 25 0 16
Steven Strogatz 2008: 0 28 14
Nicholas Negroponte 2008: 10 2 1
Jennifer 8. Lee 2008: 29 2 17
Kary Mullis 2009: 28 0 20
John Maeda 2009: 30 0 33
Paul Sereno 2009: 0 21 10
Paul Moller 2009: 0 2 26
Greg Lynn 2009: 8 0 33
Rob Forbes 2009: 33 0 15
Sasa Vucinic 2006: 2 32 0
Scott McCloud 2009: 0 27 32
Peter Reinhart 2009: 0 29 18
Joseph Pine 2009: 2 0 28
Paula Scher 2009: 33 0 16
David Carson 2009: 0 33 28
Jamais Cascio 2009: 25 0 2
Barry Schuler 2009: 31 0 21
Sherwin Nuland 2009: 24 1 0
Woody Norris 2009: 27 0 28
Peter Ward 2009: 25 0 21
Burt Rutan 2006: 0 28 2
Aimee Mullins 2009: 28 1 0
Joe DeRisi 2009: 31 6 0
Natalie MacMaster 2009: 12 16 28
Bill Gross 2009: 26 0 25
Bill Gates 2009: 10 0 2
Elizabeth Gilbert 2009: 0 28 1
Milton Glaser 2009: 0 17 28
David Merrill 2009: 12 0 16
Barry Schwartz 2009: 20 10 0
Juan Enriquez 2009: 14 11 0
Ben Saunders 2006: 13 25 1
Jose Antonio Abreu 2009: 12 10 2
Gustavo Dudamel and the Teresa Carre√±o Youth Orchestra 2009: 12 2 16
Sylvia Earle 2009: 13 25 2
Jill Tarter 2009: 4 25 0
Ed Ulbrich 2009: 0 1 23
Charles Moore 2009: 13 29 1
Richard Pyle 2009: 0 13 21
Miru Kim 2009: 15 1 8
Evan Williams 2009: 0 20 9
Brenda Laurel 2009: 16 0 10
Edward Burtynsky 2006: 0 2 15
Willie Smits 2009: 21 0 26
Nalini Nadkarni 2009: 21 0 20
Mike Rowe 2009: 28 0 1
Don Norman 2009: 18 33 0
Pattie Maes + Pranav Mistry 2009: 34 0 32
Aimee Mullins 2009: 1 10 0
Stuart Brown 2009: 16 0 7
Tim Berners-Lee 2009: 23 0 32
Dan Dennett 2009: 7 0 28
Dan Ariely 2009: 20 0 1
Hans Rosling 2006: 2 19 23
Robert Fischell 2006: 3 7 0
Adam Savage 2009: 0 1 28
Bruce McCall 2009: 1 0 28
Kamal Meattle 2009: 21 8 29
Saul Griffith 2009: 26 0 11
Jacqueline Novogratz 2009: 1 22 2
David Pogue 2009: 28 32 34
John Wooden 2009: 16 1 0
Nathan Wolfe 2009: 21 22 0
C.K. Williams 2009: 1 5 0
Jacek Utko 2009: 33 0 1
Bono 2006: 19 0 2
Ueli Gegenschatz 2009: 0 1 25
Christopher C. Deam 2009: 33 0 8
P.W. Singer 2009: 11 2 30
Nathaniel Kahn 2009: 1 8 28
Bruce Bueno de Mesquita 2009: 28 20 0
Bonnie Bassler 2009: 31 14 0
Emily Levine 2009: 28 0 1
Renny Gleeson 2009: 0 28 27
Shai Agassi 2009: 26 2 0
Gregory Stock 2009: 0 28 3
Michael Shermer 2006: 0 4 23
JoAnn Kuchera-Morin 2009: 7 23 0
Tim Ferriss 2009: 1 18 0
Matthew Childs 2009: 0 28 5
Margaret Wertheim 2009: 0 8 25
Niels Diffrient 2009: 0 1 28
Nate Silver 2009: 20 0 2
Erik Hersman 2009: 34 19 0
Ben Katchor 2009: 27 8 1
Alex Tabarrok 2009: 2 19 26
Michael Merzenich 2009: 7 17 0
Peter Donnelly 2006: 0 28 3
Sarah Jones 2009: 28 1 0
Laurie Garrett 2009: 2 0 28
Brian Cox 2009: 30 4 0
Sean Gourley 2009: 34 23 0
Mae Jemison 2009: 0 28 4
Tom Shannon 2009: 25 0 8
Al Gore 2009: 2 13 26
Louise Fresco 2009: 29 0 2
Seth Godin 2009: 20 0 28
Hans Rosling 2009: 22 19 2
Kevin Kelly 2006: 0 16 25
Nandan Nilekani 2009: 2 17 0
Naturally 7 2009: 0 27 1
Ray Anderson 2009: 26 2 25
Dan Ariely 2009: 0 20 28
Mary Roach 2009: 5 28 1
Carolyn Porco 2009: 18 25 0
Yves Behar 2009: 33 26 0
Joachim de Posada 2009: 10 28 1
Jay Walker 2009: 17 2 10
Michelle Obama 2009: 5 1 10
Ray Kurzweil 2006: 0 7 31
Jonathan Drori 2009: 21 29 2
Kaki King 2009: 4 12 28
Liz Coleman 2009: 2 10 0
Ray Kurzweil 2009: 34 7 23
Yann Arthus-Bertrand 2009: 25 1 26
Felix Dennis 2009: 1 28 5
Pete Alcorn 2009: 2 0 23
Kevin Surace 2009: 28 0 8
John La Grou 2009: 26 23 2
Nancy Etcoff 2009: 0 7 20
Peter Gabriel 2006: 1 0 2
Robert Full 2009: 11 0 21
Richard St. John 2009: 1 28 0
Jane Poynter 2009: 0 29 21
Clay Shirky 2009: 32 2 0
Diane Benscoter 2009: 7 1 0
Catherine Mohr 2009: 11 3 0
Philip Zimbardo 2009: 10 0 1
Paul Collier 2009: 2 0 8
Katherine Fulton 2009: 0 2 28
Ray Zahab 2009: 1 19 13
Dean Ornish 2006: 22 6 3
Arthur Benjamin 2009: 10 23 0
Gever Tulley 2009: 10 8 0
Daniel Libeskind 2009: 8 0 15
Eames Demetrios 2009: 33 0 28
Tom Wujec 2009: 7 0 34
Sophal Ear 2009: 1 2 0
Kary Mullis 2009: 31 28 0
Stewart Brand 2009: 26 15 2
Olafur Eliasson 2009: 0 15 8
Daniel Kraft 2009: 14 3 0
Rives 2006: 32 28 12
Jim Fallon 2009: 7 1 31
Nina Jablonski 2009: 21 0 25
Gordon Brown 2009: 2 1 20
Alain de Botton 2009: 0 20 1
Golan Levin 2009: 0 20 11
Elaine Morgan 2009: 28 18 21
Willard Wigan 2009: 1 0 28
Michael Pritchard 2009: 18 28 0
Paul Romer 2009: 15 2 20
Janine Benyus 2009: 18 0 8
Richard St. John 2006: 28 10 0
Emmanuel Jal 2009: 12 1 28
Dan Pink 2009: 0 2 20
Eric Giler 2009: 0 26 28
Hans Rosling 2009: 2 19 22
Natasha Tsakos 2009: 0 1 10
Cary Fowler 2009: 2 0 28
Joshua Silver 2009: 2 0 3
Geoff Mulgan 2009: 2 0 20
Evan Grant 2009: 27 4 0
Steve Truglia 2009: 0 25 8
Tony Robbins 2006: 28 1 20
Robert Neuwirth 2007: 15 20 8
James Balog 2009: 13 25 0
Lewis Pugh 2009: 18 1 13
Rebecca Saxe 2009: 7 20 28
Misha Glenny 2009: 2 0 1
Bjarke Ingels 2009: 0 8 18
John Lloyd 2009: 4 28 0
Oliver Sacks 2009: 7 14 1
Imogen Heap 2009: 28 25 16
Jonathan Zittrain 2009: 32 0 28
Evgeny Morozov 2009: 32 0 20
Bjorn Lomborg 2007: 2 0 28
William Kamkwamba 2009: 1 30 17
Taryn Simon 2009: 1 2 0
Jacqueline Novogratz 2009: 2 8 0
Parag Khanna 2009: 2 19 26
Tim Brown 2009: 33 0 2
Karen Armstrong 2009: 24 20 1
Garik Israelian 2009: 25 4 0
Stefan Sagmeister 2009: 0 33 1
Carolyn Steel 2009: 29 15 0
David Logan 2009: 20 0 28
Phil Borges 2007: 1 10 20
Chimamanda Ngozi Adichie 2009: 1 19 17
Beau Lotto 2009: 7 0 34
Sam Martin 2009: 8 0 4
Eric Sanderson 2009: 15 21 0
David Hanson 2009: 11 30 0
Rory Sutherland 2009: 0 29 20
Henry Markram 2009: 7 4 0
Julian Treasure 2009: 27 12 0
John Gerzema 2009: 0 2 15
Paul Debevec 2009: 0 30 34
Wade Davis 2007: 21 1 17
Itay Talgam 2009: 12 28 0
Marc Koska 2009: 22 34 0
Ian Goldin 2009: 2 0 25
David Deutsch 2009: 4 0 25
Rachel Armstrong 2009: 8 15 0
Becky  Blanton 2009: 1 0 20
Marcus du Sautoy 2009: 17 0 4
Stefana Broadbent 2009: 0 10 20
Cameron Sinclair 2009: 2 8 26
Rachel Pike 2009: 0 2 25
Martin Rees 2007: 25 4 0
Edward Burtynsky 2009: 26 0 1
Cynthia Schneider 2009: 5 12 0
Pranav Mistry 2009: 0 9 34
Mathieu Lehanneur 2009: 33 0 7
Fields Wicker-Miurin 2009: 22 2 21
Devdutt Pattanaik 2009: 1 2 0
Tom Wujec 2009: 0 4 25
Hans Rosling 2009: 2 3 1
Rob Hopkins 2009: 26 0 29
Magnus Larsson 2009: 8 0 2
Robert Wright 2007: 28 0 16
Mallika Sarabhai 2009: 1 18 5
Shashi Tharoor 2009: 2 1 34
Gordon Brown 2009: 2 9 20
Andrea Ghez 2009: 4 0 25
Anupam Mishra 2009: 18 17 0
Scott Kim 2009: 16 0 28
Sunitha Krishnan 2009: 1 22 5
Rory  Bremner 2009: 28 1 2
Marc Pachter 2009: 1 28 20
Thulasiraj Ravilla 2009: 3 0 2
Steven Johnson 2007: 15 18 0
Shereen El Feki 2009: 12 0 1
Loretta Napoleoni 2009: 2 1 0
Ryan Lobo 2009: 5 1 24
Alexis Ohanian 2009: 32 0 28
Charles Anderson 2009: 13 19 18
James Geary 2009: 17 0 27
Shaffi Mather 2009: 2 9 0
Steven  Cowley 2009: 26 0 28
Asher Hasan 2009: 2 1 12
Bertrand Piccard 2010: 26 0 25
Charles Leadbeater 2007: 0 32 2
Vilayanur Ramachandran 2010: 7 0 14
Nick Veasey 2010: 0 30 33
Dan Buettner 2010: 1 0 29
Romulus Whitaker 2010: 21 18 0
Herbie Hancock 2010: 28 1 34
Kartick Satyanarayan 2010: 10 2 1
Kiran Sethi 2010: 10 15 1
Lalitesh Katragadda 2010: 2 15 19
Charles Fleischer 2010: 4 28 0
David Blaine 2010: 0 1 18
Anna Deavere Smith 2007: 28 1 0
Ravin Agrawal 2010: 0 15 5
Anthony Atala 2010: 14 3 0
Bill Davenhall 2010: 3 0 34
Joshua Prince-Ramus 2010: 8 0 20
Eve Ensler 2010: 5 1 14
Jane Chen 2010: 3 2 0
Derek Sivers 2010: 12 28 1
Sendhil Mullainathan 2010: 0 26 2
Jamie Heywood 2010: 3 23 22
George Whitesides 2010: 0 34 3
Saul Griffith 2007: 4 0 30
David Agus 2010: 6 0 3
Tom Shannon, John Hockenberry 2010: 4 0 25
Peter Eigen 2010: 2 0 19
Jamie Oliver 2010: 29 10 28
Blaise Agüera y Arcas 2010: 0 15 28
David Cameron 2010: 34 2 0
Aimee Mullins 2010: 1 10 0
Bill Gates 2010: 26 0 9
Kevin Kelly 2010: 0 26 29
Philip K. Howard 2010: 2 0 10
Joshua Prince-Ramus 2006: 8 0 33
Neil Gershenfeld 2007: 30 32 0
Eric Topol 2010: 3 0 6
Temple Grandin 2010: 10 28 0
Pawan Sinha 2010: 3 10 0
Raghava KK 2010: 1 28 10
Daniel Kahneman 2010: 0 3 1
Harsha Bhogle 2010: 16 2 1
Gary Flake 2010: 34 23 0
James Cameron 2010: 13 0 1
The LXD 2010: 12 0 28
Srikumar Rao 2010: 0 1 28
Carl Honoré 2007: 0 1 28
Tim Berners-Lee 2010: 23 0 15
Gary Lauder's new traffic sign 2010: 26 0 2
Dan Barber 2010: 13 29 28
Eric Mead 2010: 0 28 20
Mark Roth 2010: 0 1 21
Eric Dishman 2010: 3 0 23
Jane McGonigal 2010: 16 0 2
Ken Kamler 2010: 7 26 1
Shekhar Kapur 2010: 1 4 28
Sam Harris 2010: 0 24 5
E.O. Wilson 2007: 21 31 25
Juliana Machado Ferreira 2010: 21 2 0
Alan Siegel 2010: 17 2 0
Joel Levine 2010: 25 18 0
Robert Gupta 2010: 12 1 0
Kevin Bales 2010: 2 20 1
Shukla Bose 2010: 10 1 17
Kirk Citron 2010: 11 2 31
Derek Sivers 2010: 0 28 20
Adora Svitak 2010: 10 0 28
Elizabeth Pisani 2010: 22 0 3
James Nachtwey 2007: 1 2 19
Dean Kamen 2010: 28 0 1
Dennis Hong 2010: 11 0 26
Jonathan Drori 2010: 21 0 14
Natalie Merchant 2010: 1 28 12
Michael Specter 2010: 28 29 20
Jonathan Klein 2010: 0 22 2
Catherine Mohr 2010: 26 18 0
Thelma Golden 2010: 0 1 8
Edith Widder 2010: 13 0 21
James Randi 2010: 28 20 0
Bill Clinton 2007: 2 3 20
Frederick Balagadde 2010: 22 19 3
Tom Wujec 2010: 0 33 10
Omar Ahmad 2010: 0 21 28
Kavita Ramdas 2010: 5 1 12
Stephen Wolfram 2010: 4 0 17
Roz Savage 2010: 13 0 1
George Whitesides 2010: 0 9 32
Sebastian Wernicke 2010: 0 28 23
Esther Duflo 2010: 10 20 2
Simon Sinek 2010: 20 7 0
Chris Bangle 2007: 28 0 33
Jeremy Jackson 2010: 13 0 21
Anil Gupta 2010: 2 20 30
Thomas Dolby 2010: 1 8 26
Nicholas Christakis 2010: 20 0 32
Nathan Myhrvold 2010: 0 14 3
Enric Sala 2010: 13 2 0
Dan Meyer 2010: 10 0 34
Julia Sweeney 2010: 28 5 1
William Li 2010: 6 0 14
Graham Hill 2010: 21 3 1
Craig Venter 2007: 31 0 13
Dee Boersma 2010: 0 26 18
Richard Sears 2010: 26 0 28
Craig Venter 2010: 31 0 14
Ken Robinson 2010: 10 1 0
Johanna Blakley 2010: 0 33 28
Sharmeen Obaid-Chinoy 2010: 10 1 16
Seth Berkley 2010: 22 14 0
Lawrence Lessig 2010: 0 12 32
John Underkoffler 2010: 0 9 30
Brian Skerry 2010: 13 21 1
Dean Kamen 2007: 15 26 0
Christopher "moot" Poole" 2010: 9 32 28
Brian Cox 2010: 25 4 0
Adam Sadowsky 2010: 30 12 0
Michael Sandel 2010: 16 0 2
John Kasaona 2010: 19 1 2
Rory Sutherland 2010: 0 2 20
Stewart Brand + Mark Z. Jacobson 2010: 26 9 0
David Byrne 2010: 12 0 27
Michael Shermer 2010: 0 7 28
Margaret Gould Stewart 2010: 32 16 0
Jane Goodall 2007: 21 0 24
Peter Tyack 2010: 27 13 0
Cameron Herold 2010: 10 28 0
Ananda Shankar Jayant 2010: 6 1 26
Chip Conley 2010: 0 2 1
Marian Bantjes 2010: 0 33 1
Charles Leadbeater 2010: 10 2 0
Aditi Shankardass 2010: 7 10 3
Hillel Cooperman 2010: 28 30 0
Clay Shirky 2010: 0 32 33
Ellen Dunham-Jones 2010: 8 0 15
Golan Levin 2007: 30 0 27
Stephen Palumbi 2010: 13 0 29
Carter Emmart 2010: 4 25 0
Mitchell Joachim 2010: 14 8 0
Benoit Mandelbrot 2010: 0 1 28
Ellen  Gustafson 2010: 29 0 2
Nalini Nadkarni 2010: 21 0 5
Hans Rosling 2010: 2 10 1
Carl Safina 2010: 26 13 18
Matt Ridley 2010: 0 21 20
Ethan Zuckerman 2010: 32 0 2
Julia Sweeney 2006: 1 28 24
Janine Benyus 2007: 18 0 33
Elif Shafak 2010: 1 17 5
Julian Assange 2010: 9 34 0
Naif Al-Mutawa 2010: 1 28 0
Dimitar Sasselov 2010: 25 4 0
Tan Le 2010: 7 0 30
Kevin Stone 2010: 3 0 14
Sheena Iyengar 2010: 1 0 20
John Delaney 2010: 13 25 0
Laurie Santos 2010: 0 28 21
Lewis Pugh 2010: 18 1 25
Seth Godin 2007: 20 0 28
Jason Clay 2010: 2 25 28
Sheryl WuDunn 2010: 10 5 2
Peter Molyneux 2010: 0 27 1
Jamil Abu-Wardeh 2010: 1 0 32
Maz Jobrani 2010: 28 2 1
Seth Priebatsch 2010: 16 0 20
David McCandless 2010: 23 34 0
Lee Hotz 2010: 25 13 2
Jim Toomey 2010: 13 0 1
Lisa Margonelli 2010: 26 0 18
Thom Mayne 2007: 8 0 15
Dan Cobley 2010: 4 0 23
Nic Marks 2010: 2 25 0
Johan Rockstrom 2010: 25 2 26
His Holiness the Karmapa 2010: 0 1 20
Derek Sivers 2010: 0 1 28
Rachel Sussman 2010: 21 0 1
Sugata Mitra 2010: 10 1 28
Alwar Balasubramaniam 2010: 28 0 1
Carne Ross 2010: 2 1 0
Ben Cameron 2010: 32 2 0
Vik Muniz 2007: 0 28 1
Rob Dunbar 2010: 13 25 0
Chris Anderson 2010: 32 0 16
Jessa Gamble 2010: 0 14 21
Nicholas Christakis 2010: 20 0 23
Caroline Phillips 2010: 12 27 16
Christien Meindertsma 2010: 0 3 29
Steven Johnson 2010: 0 8 28
Mitchell Besser 2010: 22 3 19
Annie Lennox 2010: 22 19 1
Fabian Hemmert 2010: 0 34 32
James Watson 2007: 31 28 1
Julian Treasure 2010: 27 12 0
Gary Wolf 2010: 23 0 3
Sebastian Seung 2010: 7 0 18
Inge Missmahl 2010: 3 1 20
Mechai Viravaidya 2010: 10 2 5
Eben Bayer 2010: 0 26 8
Tim Jackson 2010: 2 0 26
Barbara Block 2010: 13 23 0
Hans Rosling 2010: 19 2 10
Stacey Kramer 2010: 0 1 7
Frans Lanting 2007: 25 18 1
Stefano Mancuso 2010: 21 29 7
Melinda Gates 2010: 2 3 20
Peter Haas 2010: 8 2 29
Natalie Jeremijenko 2010: 3 0 18
Ze Frank 2010: 28 1 0
Jessica Jackley 2010: 0 1 2
Heribert Watzke 2010: 7 29 26
Dianna Cohen 2010: 0 13 29
Patrick Chappatte 2010: 32 19 1
David Byrne, Ethel + Thomas Dolby 2010: 28 1 21
Paul Bennett 2007: 0 33 3
R.A. Mashelkar 2010: 2 24 0
Joseph Nye 2010: 2 26 0
Barton Seaver 2010: 29 13 0
Shimon Steinberg 2010: 0 21 29
Miwa Matreyek 2010: 4 1 13
Tom Chatfield 2010: 16 0 20
David Bismark 2010: 0 20 23
Greg Stone 2010: 13 2 1
Gero Miesenboeck 2010: 7 14 0
Andrew Bird 2010: 27 28 0
Nick Bostrom 2007: 0 20 21
Emily Pilloton 2010: 33 10 0
Stefan Wolff 2010: 2 33 1
Aaron Huey 2010: 2 1 0
Auret van Heerden 2010: 2 0 28
Eric Berlow 2010: 21 0 32
Conrad Wolfram 2010: 0 10 30
Denis Dutton 2010: 0 21 12
Shimon Schocken 2010: 10 1 0
John Hardy 2010: 10 8 1
Kristina Gjerde 2010: 13 0 2
Stefan Sagmeister 2007: 33 0 28
Kim Gorgens 2010: 10 28 16
Zainab Salbi 2010: 5 1 2
Jason Fried 2010: 0 20 28
Dan Phillips 2010: 28 0 8
Birke Baehr 2010: 29 28 31
William Ury 2010: 1 2 0
Marcel Dicke 2010: 29 21 2
Bart Weetjens 2010: 29 0 22
Arthur Potts Dawson 2010: 29 18 0
Halla Tomasdottir 2010: 5 2 0
Alex Steffen 2007: 0 15 25
Tony Porter 2010: 5 1 28
Kiran Bedi 2010: 1 10 28
Hanna Rosin 2010: 5 0 1
Diana Laufenberg 2010: 10 34 0
Rufus Griscom + Alisa Volkman 2010: 1 0 28
Rachel Botsman 2010: 0 32 2
Beverly + Dereck Joubert 2010: 0 19 1
Sheryl Sandberg 2010: 5 28 0
Majora Carter 2010: 2 0 26
Brené Brown 2010: 28 1 20
Rick Warren 2006: 28 1 0
Susan Savage-Rumbaugh 2007: 17 21 0
Barry Schwartz 2010: 10 20 0
Arianna Huffington 2011: 5 0 1
Lesley Hazleton 2011: 1 17 5
Charles Limb 2011: 7 12 0
Deborah Rhodes 2011: 6 5 0
Neil Pasricha 2011: 1 28 0
Jody Williams 2011: 2 5 28
Amber Case 2011: 0 32 1
Thomas Thwaites 2011: 0 18 1
Elizabeth Lesser 2011: 0 1 27
Sheila Patek 2007: 0 21 27
Ali Carr-Chellman 2011: 10 16 5
Naomi Klein 2011: 26 0 25
Charity Tillemann-Dick 2011: 1 3 28
Van Jones 2011: 20 6 21
Anders Ynnerman 2011: 23 0 30
Heather Knight 2011: 11 23 0
Martin Jacques 2011: 2 0 17
Thomas Goetz 2011: 34 3 0
Liza Donnelly 2011: 5 0 28
Ariel Garten 2011: 0 16 7
Al Seckel 2007: 0 28 33
Bruce Feiler 2011: 1 28 6
Kate Orff 2011: 18 15 0
Dale Dougherty 2011: 0 28 8
Johanna Blakley 2011: 32 5 0
Christopher McDougall 2011: 0 5 1
Suheir Hammad 2011: 1 5 28
Nigel Marsh 2011: 1 0 10
Cynthia Breazeal 2011: 11 0 16
Hawa Abdi + Deqo Mohamed 2011: 5 3 1
Michael Pawlyn 2011: 18 26 0
Juan Enriquez 2007: 31 0 14
Geert Chatrou 2011: 12 28 0
Krista Tippett 2011: 24 0 4
Patricia Kuhl 2011: 17 7 27
Jacqueline Novogratz 2011: 5 1 2
Lisa Gansky 2011: 0 15 23
Madeleine Albright 2011: 5 2 0
Noreena Hertz 2011: 0 3 2
Iain Hutchison 2011: 6 3 0
Elizabeth Lindsey 2011: 1 5 25
Danny Hillis 2011: 6 0 3
Nora York 2007: 28 3 2
Ahn Trio 2011: 12 0 3
Wadah Khanfar 2011: 20 2 1
JR 2011: 1 5 0
Wael Ghonim 2011: 20 32 0
Bill Gates 2011: 2 0 3
Anthony Atala 2011: 14 3 0
Courtney Martin 2011: 5 1 0
Salman Khan 2011: 10 0 28
Deb Roy 2011: 17 18 23
Rob Harmon 2011: 18 2 28
Jill Sobule 2007: 28 1 21
David Brooks 2011: 0 10 7
Janna Levin 2011: 4 27 25
Mark Bezos 2011: 1 28 29
Rogier van der Heide 2011: 8 4 0
Sarah Kay 2011: 1 28 0
Hans Rosling 2011: 30 26 20
Isabel Behncke 2011: 16 21 0
Paul Root Wolpe 2011: 31 21 0
Eythor Bender 2011: 11 0 30
Claron McFadden 2011: 27 1 0
Caroline Lavelle 2007: 28 1 25
Patricia Ryan 2011: 17 10 0
Ralph Langner 2011: 0 23 29
Handspring Puppet Co. 2011: 0 1 19
Sebastian Thrun 2011: 15 1 0
Eric Whitacre 2011: 1 12 0
AnnMarie Thomas 2011: 16 10 0
Stanley McChrystal 2011: 0 1 28
Chade-Meng Tan 2011: 24 0 2
Morgan Spurlock 2011: 0 1 28
Mick Ebeling 2011: 1 28 0
Dan Dennett 2007: 0 28 7
Caroline Casey 2011: 28 1 0
Jackson Browne 2011: 2 0 28
David Christian 2011: 4 31 34
Dave Meslin 2011: 0 34 20
Roger Ebert 2011: 27 30 1
Marcin Jakubowski 2011: 27 0 8
Susan Lim 2011: 14 1 2
Sam Richards 2011: 28 2 1
Kathryn Schulz 2011: 0 28 1
John Hunter 2011: 16 10 1
Evelyn Glennie 2007: 12 27 0
Anil Ananthaswamy 2011: 4 18 0
Ric Elias 2011: 1 0 28
Harvey Fineberg 2011: 31 14 0
Bruce Schneier 2011: 0 2 20
Angela Belcher 2011: 31 14 0
Mike Matas 2011: 26 9 32
Arvind Gupta 2011: 10 0 27
Eli Pariser 2011: 32 34 0
Aicha el-Wafi + Phyllis Rodriguez 2011: 1 5 28
Carlo Ratti 2011: 0 8 18
William McDonough 2007: 26 15 8
Suzanne Lee 2011: 0 18 29
Sean Carroll 2011: 4 0 28
Louie Schwartzberg 2011: 1 21 4
Paul Nicklen 2011: 13 25 1
Fiorenzo Omenetto 2011: 18 0 22
Ron Gutman 2011: 7 0 1
Amit Sood 2011: 0 28 32
Leonard Susskind 2011: 1 28 0
Ed Boyden 2011: 14 7 0
Thomas Heatherwick 2011: 8 0 26
Dan Dennett 2006: 33 0 28
Jeff Bezos 2007: 32 0 28
Elliot Krane 2011: 14 3 7
Edith Widder 2011: 13 0 4
Terry Moore 2011: 1 0 28
Aaron Koblin 2011: 0 12 20
Bruce Aylward 2011: 2 0 10
Shirin Neshat 2011: 5 2 1
Mustafa Akyol 2011: 2 0 5
Robert Gupta + Joshua Roman 2011: 34 8 14
Dennis Hong 2011: 34 0 30
Stefan Sagmeister 2011: 0 1 21
Rives 2007: 28 15 27
Aaron O'Connell 2011: 0 7 16
Jessi Arrington 2011: 28 0 20
Damon Horowitz 2011: 28 23 0
Jack Horner 2011: 31 0 14
Janet Echelman 2011: 15 8 1
Paul Romer 2011: 15 2 8
Alice Dreger 2011: 0 20 5
JD Schramm 2011: 1 0 20
Daniel Kraft 2011: 3 14 0
Shea Hembrey 2011: 0 1 8
Eddi Reader 2007: 27 0 28
Steve Keil 2011: 16 0 28
Camille Seaman 2011: 18 25 1
Onyx Ashanti 2011: 27 12 33
Maya Beiser 2011: 12 27 4
Bill Ford 2011: 0 2 15
Daniel Tammet 2011: 17 0 27
Jok Church 2011: 1 28 10
Honor Harger 2011: 27 4 25
Joshua Walters 2011: 28 0 22
Emiliano Salinas 2011: 2 1 28
Eddi Reader 2007: 15 28 1
Rajesh Rao 2011: 17 0 30
Dave deBronkart 2011: 3 6 23
Robert Hammond 2011: 15 8 0
Matt Cutts 2011: 1 0 17
Nathan Myhrvold 2011: 0 29 18
Jonathan Drori 2011: 21 29 0
Simon Lewis 2011: 7 1 0
Nina Tandon 2011: 14 29 0
Rebecca MacKinnon 2011: 32 2 0
Maajid Nawaz 2011: 2 0 1
Tom Honey 2007: 24 1 4
Tim Harford 2011: 0 3 2
Nadia Al-Sakkaf 2011: 5 1 0
Mikko Hypponen 2011: 32 0 30
Thandie Newton 2011: 0 1 19
Kevin Slavin 2011: 0 28 15
Markus Fischer 2011: 26 0 8
Rory Stewart 2011: 2 1 0
Geoffrey West 2011: 15 2 0
Paul Bloom 2011: 0 20 1
Josette Sheeran 2011: 29 2 10
Richard Dawkins 2007: 4 0 2
Julian Treasure 2011: 27 0 1
Adam Ostrow 2011: 32 0 6
Harald Haas 2011: 23 4 26
Mark Pagel 2011: 17 21 0
Jessica Green 2011: 8 23 3
Philip Zimbardo 2011: 5 32 10
Eve Ensler 2011: 6 1 5
Alex Steffen 2011: 26 15 0
Dyan deNapoli 2011: 26 1 13
Jeremy Gilley 2011: 1 0 2
Tom Rielly 2007: 28 0 33
Lucianne Walkowicz 2011: 25 4 0
Marco Tempest 2011: 5 1 28
Dan Ariely 2011: 28 23 0
Svante Pääbo 2011: 19 31 0
Julia Bacha 2011: 2 1 0
Skylar Tibbits 2011: 8 0 31
Joan Halifax 2011: 24 5 1
Edward Tenner 2011: 0 30 3
Sarah Kaminsky 2011: 1 0 19
Lee Cronin 2011: 0 31 4
Rachelle Garniez 2007: 9 34 8
Raghava KK 2011: 10 1 28
Yasheng Huang 2011: 2 5 23
Misha Glenny 2011: 32 2 0
Kate Hartman 2011: 0 27 8
Richard Resnick 2011: 6 0 31
Lauren Zalaznick 2011: 0 2 1
Niall Ferguson 2011: 2 0 28
Jean-Baptiste Michel + Erez Lieberman Aiden 2011: 0 23 17
Amy Lockwood 2011: 22 20 2
Elizabeth Murchison 2011: 6 14 31
Chris Anderson 2007: 0 2 32
Sunni Brown 2011: 34 0 17
Abraham Verghese 2011: 3 1 6
Geoff Mulgan 2011: 10 0 2
Jarreth Merz 2011: 19 1 2
Ben Goldacre 2011: 6 23 22
Danielle de Niese 2011: 28 4 1
Yang Lan 2011: 2 32 20
Christoph Adami 2011: 0 25 1
Graham Hill 2011: 0 8 28
Mike Biddle 2011: 0 26 2
Natalie MacMaster 2007: 13 18 25
Charles Hazlewood 2011: 12 0 19
Alison Gopnik 2011: 0 10 7
Richard Seymour 2011: 0 28 33
Ian Ritchie 2011: 30 34 32
Pamela Meyer 2011: 28 17 0
Jae Rhim Lee 2011: 29 0 6
Bunker Roy 2011: 1 5 10
Justin Hall-Tipping 2011: 26 0 18
Guy-Philippe Goldstein 2011: 2 0 8
Todd Kuiken 2011: 0 3 7
```

```python
>>> doctopic.shape
(2113, 35)
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

## Topics by Year

I have 35 topics, so 35 boxes: each with 10 bars.
