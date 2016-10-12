
# TEDtalk NMF Topics

## Preliminaries


```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib

See also this for suggestions: http://scikit-learn.org/dev/auto_examples/applications/topics_extraction_with_nmf_lda.html#example-applications-topics-extraction-with-nmf-lda-py

More useful discussion of NMF-LDA on this [SO thread][so].

[so]: http://stackoverflow.com/questions/35140117/how-to-interpret-lda-components-using-sklearn


```python
import pandas
import re
colnames = ['author', 'title', 'date' , 'length', 'text']
data = pandas.read_csv('./data/talks-v1b.csv', names=colnames)

# Creating 3 lists of relevant data.
# Importing everything here. 
# If we want to test, we should import 2006-2015 and test on 2016.

talks = data.text.tolist()
authors = data.author.tolist()
dates = data.date.tolist()

# Getting only the years from dates list
years = [re.sub('[A-Za-z ]', '', item) for item in dates]

# Combining year with presenter for citation
authordate = [author+" "+year for author, year in zip(authors, years)]

# We need to remove the "empty" talks from both lists.

# We establish which talks are empty
i = 0
no_good = []
for talk in talks: 
    A = type(talk)
    B = type('string or something')
    if A != B:
        no_good.append(i)
    i = i + 1

# Now we delete them in reverse order so as to preserve index order
for index in sorted(no_good, reverse=True):
    del talks[index]
for index in sorted(no_good, reverse=True):
    del authordate[index]
```

## Non-Negative Matrix Topic Models

The block of code below produces a list saved as an `np.array`, of 7254 words and a **document term matrix**, `dtm.shape`, of `(2106, 7254)`. The `CountVectorizer` does a lot of work, and it has the following parameters:

* `stop_words` specifies which set to use. (The English words are the same as the Glasgow Information Retrieval Group. See link on [GitHub][].)
* `lowercase` (default `True`) convert all text to lowercase before tokenizing
* `min_df` (default 1) remove terms from the vocabulary that occur in fewer than min_df documents (in a large corpus this may be set to 15 or higher to eliminate very rare words)
vocabulary ignore words that do not appear in the provided list of words
* `token_pattern` (default `u'(?u)\b\w\w+\b'`) regular expression identifying tokens–by default words that consist of a single character (e.g., ‘a’, ‘2’) are ignored, setting `token_pattern` to `'(?u)\b\w+\b'` will include these tokens
* `tokenizer` (default unused) use a custom function for tokenizing

[GitHub]: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/stop_words.py


```python
import numpy as np
import sklearn.feature_extraction.text as text
from sklearn.decomposition import NMF, LatentDirichletAllocation

# Function for printing topic words (used later):
def print_top_words(model, feature_names, n_top_words):
    for topic_id, topic in enumerate(model.components_):
        print('\nTopic {}:'.format(int(topic_id))) 
        print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
              +', ' for i in topic.argsort()[:-n_top_words - 1:-1]]))

# Show the top X words in a topic:
#for t in range(len(topic_words)):
#    print("Topic {}: {}".format(t, ' '.join(topic_words[t][:15])))
    
n_samples = len(talks)
n_features = 1000
n_topics = 35
n_top_words = 20

# Use tf-idf features for NMF.
tfidf_vectorizer = text.TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(talks)
tf_vectorizer = text.CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')

# Use tf (raw term count) features for LDA.
tf = tf_vectorizer.fit_transform(talks)
```


```python
# Fit the NMF model
print("Fitting the NMF model with tf-idf features, "
      "n_samples={} and n_features={}...".format(n_samples, n_features))
nmf = NMF(n_components=n_topics, 
          random_state=1,
          alpha=.1, 
          l1_ratio=.5).fit(tfidf)

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

# Scale component values so that they add up to 1 for any given document
#doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
```


```python
# Now to associate NMF topics to documents...

dtm = tf.toarray()
doctopic = nmf.fit_transform(dtm)

print("Top NMF topics in...")
for i in range(len(doctopic)):
    top_topics = np.argsort(doctopic[i,:])[::-1][0:3]
    top_topics_str = ' '.join(str(t) for t in top_topics)
    print("{}: {}".format(authordate[i], top_topics_str))
```

    Top NMF topics in...
    Al Gore 2006: 8 2 3
    David Pogue 2006: 9 13 2
    Cameron Sinclair 2006: 9 34 23
    Sergey Brin + Larry Page 2007: 0 5 13
    Nathalie Miebach 2011: 10 0 23
    Richard Wilkinson 2011: 10 1 20
    Malcolm Gladwell 2011: 1 0 17
    Jay Bradner 2011: 32 27 5
    Béatrice Coron 2011: 3 34 25
    Hasan Elahi 2011: 13 26 0
    Paul Zak 2011: 1 2 4
    Anna Mracek Dietrich 2011: 13 0 3
    Daniel Wolpert 2011: 4 8 15
    Marco Tempest 2011: 31 18 8
    Stew 2007: 6 33 9
    Martin Hanczyc 2011: 22 25 26
    Aparna Rao 2011: 23 1 33
    Ben Kacyra 2011: 31 3 23
    Allan Jones 2011: 4 10 27
    Charlie Todd 2011: 1 34 3
    Alexander Tsiaras 2011: 26 27 8
    Yves Rossy 2011: 0 13 25
    Thomas Suarez 2011: 12 25 9
    Cynthia Kenyon 2011: 0 13 27
    Robin Ince 2011: 33 16 7
    James Howard Kunstler 2007: 8 34 5
    Phil Plait 2011: 19 8 13
    Péter Fankhauser 2011: 31 18 25
    Joe Sabia 2011: 3 14 1
    Britta Riley 2011: 24 26 34
    Amy Purdy 2011: 22 3 8
    Damon Horowitz 2011: 5 18 13
    Annie Murphy Paul 2011: 6 14 24
    John Bohannon 2011: 27 31 33
    Charles Limb 2011: 29 0 15
    Kathryn Schulz 2011: 16 22 15
    David Kelley 2007: 25 1 16
    Srdja Popovic 2011: 1 14 5
    Luis von Ahn 2011: 1 31 0
    Cheryl Hayashi 2011: 25 13 0
    Yoav Medan 2011: 4 32 22
    Stefon Harris 2011: 29 8 0
    Monika Bulaj 2011: 1 5 12
    Quyen Nguyen 2011: 32 5 11
    Pavan Sukhdev 2011: 17 20 21
    Homaro Cantu + Ben Roche 2011: 24 0 26
    Ramona Pierson 2011: 16 8 3
    Stewart Brand 2007: 8 34 1
    Antonio Damasio 2011: 4 8 7
    Sheila Nirenberg 2011: 4 13 0
    Daniel Goldstein 2011: 1 8 13
    Karen Tse 2011: 2 1 3
    Alberto Cairo 2011: 2 1 3
    AJ Jacobs 2012: 13 0 3
    Jane Fonda 2012: 22 1 3
    Paddy Ashdown 2012: 14 20 17
    Sebastian Wernicke 2012: 26 13 1
    Lauren Hodge, Shree Bose + Naomi Shah 2012: 32 28 27
    Jeff Hawkins 2007: 4 8 5
    Jonas Gahr Støre 2012: 11 8 20
    Drew Berry 2012: 27 9 0
    Morley 2012: 6 1 2
    Sonaar Luthra 2012: 21 11 1
    Alain de Botton 2012: 11 7 14
    Mikko Hypponen 2012: 20 31 18
    Clay Shirky 2012: 15 16 31
    Sheena Iyengar 2012: 1 8 26
    Bilal Bomani 2012: 21 24 11
    Julian Baggini 2012: 7 16 25
    Tierney Thys 2007: 13 25 19
    Lisa Harouni 2012: 10 31 26
    Diana Nyad 2012: 8 2 3
    Brian Goldman 2012: 3 28 25
    Gayle Tzemach Lemmon 2012: 6 14 2
    Mick Mountz 2012: 2 13 7
    Peter van Uhm 2012: 14 3 20
    Bill Doyle 2012: 32 27 22
    Shawn Achor 2012: 4 9 13
    Erica Frenkel 2012: 28 13 9
    Danny Hillis 2012: 31 8 25
    Blaise Agüera y Arcas 2007: 0 10 31
    Mike deGruy 2012: 21 8 0
    Neil Burgess 2012: 27 4 25
    Stephen Coleman 2012: 26 1 8
    Sheikha Al Mayassa 2012: 1 34 15
    Jack Horner 2012: 25 26 13
    Erik Johansson 2012: 0 7 33
    Drew Dudley 2012: 2 9 3
    Tyrone Hayes + Penelope Jagessar Chaffer 2012: 32 6 21
    Jenna McCarthy 2012: 9 1 6
    Lucien Engelen 2012: 28 25 9
    John Doerr 2007: 17 0 8
    Simon Berrow 2012: 26 13 21
    Paul Conneally 2012: 1 10 11
    Garth Lenz 2012: 14 19 13
    Neil MacGregor 2012: 14 1 3
    Chris Bliss 2012: 14 3 0
    Shilo Shiv Suleman 2012: 26 14 31
    Shlomo Benartzi 2012: 8 1 17
    Tan Le 2012: 22 3 6
    Avi Rubin 2012: 31 8 1
    Kevin Allocca 2012: 34 26 13
    Ngozi Okonjo-Iweala 2007: 30 1 17
    Paul Snelgrove 2012: 19 10 0
    Daniel Pauly 2012: 1 3 7
    Paul Gilding 2012: 11 13 24
    Peter Diamandis 2012: 17 21 14
    Vijay Kumar 2012: 15 31 25
    Susan Cain 2012: 1 11 3
    Bryan Stevenson 2012: 2 16 1
    Andrew Stanton 2012: 3 13 15
    James Hansen 2012: 19 17 14
    Jennifer Pahlka 2012: 1 23 11
    Jehane Noujaim 2006: 1 14 5
    Anand Agarawala 2007: 25 5 16
    Improv Everywhere 2012: 13 31 8
    Larry Smith 2012: 8 15 0
    Jonathan Haidt 2012: 1 13 8
    Rob Reid 2012: 13 1 17
    Brené Brown 2012: 2 8 6
    T. Boone Pickens 2012: 17 8 9
    Noel Bairey Merz 2012: 6 32 8
    Taylor Wilson 2012: 17 14 9
    Billy Collins 2012: 3 13 25
    Peter Saul 2012: 1 0 7
    Robert Thurman 2007: 7 1 22
    Donald Sadoway 2012: 17 31 23
    Regina Dugan 2012: 16 3 13
    Leymah Gbowee 2012: 12 6 2
    Ayah Bdeir 2012: 14 25 9
    Marco Tempest 2012: 3 15 1
    Sherry Turkle 2012: 1 15 31
    Chip Kidd 2012: 8 3 2
    Jack Choi 2012: 8 13 18
    Lucy McRae 2012: 31 32 9
    Jonathan Foley 2012: 21 24 14
    David Rockwell 2007: 34 7 1
    Frank Warren 2012: 1 2 9
    Frans de Waal 2012: 8 24 25
    Melinda Gates 2012: 6 12 30
    Tal Golesworthy 2012: 9 0 23
    Abigail Washburn 2012: 20 8 13
    Atul Gawande 2012: 28 1 14
    Drew Curtis 2012: 26 25 11
    Taryn Simon 2012: 23 20 6
    Laura Carstensen 2012: 1 22 3
    Christina Warinner 2012: 28 31 0
    Thomas Barnett 2007: 8 9 11
    Brian Greene 2012: 33 13 17
    Michael Norton 2012: 1 2 16
    Eduardo Paes 2012: 34 1 8
    Nancy Lublin 2012: 12 10 0
    Joe Smith 2012: 1 0 18
    Brenda Romero 2012: 29 5 2
    Liz Diller 2012: 33 0 34
    Amory Lovins 2012: 17 11 13
    Reuben Margolin 2012: 13 26 18
    Gary Kovacs 2012: 10 14 26
    Stephen Lawler 2007: 10 11 7
    Rory Sutherland 2012: 26 7 1
    Tavi Gevinson 2012: 6 7 1
    Michael Tilson Thomas 2012: 29 31 0
    JP Rangaswami 2012: 24 10 3
    Karen Bass 2012: 0 34 3
    Joshua Foer 2012: 0 13 3
    Renny Gleeson 2012: 26 16 7
    Bart Knols 2012: 13 21 1
    Tali Sharot 2012: 1 8 4
    Jean-Baptiste Michel 2012: 7 26 13
    Hans Rosling 2007: 20 28 14
    David Kelley 2012: 0 1 8
    Carl Schoonover 2012: 4 27 31
    JR 2012: 1 8 14
    Michael McDaniel 2012: 26 1 13
    Nathan Wolfe 2012: 22 19 10
    Hans Rosling 2012: 12 14 20
    Philippe Petit 2012: 3 1 14
    Shereen El-Feki 2012: 1 28 11
    Reggie Watts 2012: 5 15 33
    David MacKay 2012: 17 11 1
    Bill Stone 2007: 8 33 19
    Ken Goldberg 2012: 31 2 23
    William Noel 2012: 10 25 1
    Dalia Mogahed 2012: 6 1 20
    Sebastian Deterding 2012: 22 1 31
    Seth Shostak 2012: 5 8 18
    David Birch 2012: 5 8 9
    Juan Enriquez 2012: 27 16 0
    Diane Kelly 2012: 0 8 13
    Terry Moore 2012: 7 31 13
    Dan Dennett 2007: 13 4 1
    Damian Palin 2012: 21 13 17
    John Hodgman 2012: 34 5 4
    Ami Klin 2012: 1 16 4
    John Hockenberry 2012: 18 2 22
    Rebecca Onie 2012: 28 13 24
    Beeban Kidron 2012: 12 3 1
    Sarah Parcak 2012: 34 19 10
    LZ Granderson 2012: 1 9 13
    Rodney Mullen 2012: 18 16 9
    Megan Kamerick 2012: 6 3 13
    Alan Russell 2007: 27 8 28
    David R. Dow 2012: 1 8 12
    Ivan Oransky 2012: 26 6 15
    Marco Tempest 2012: 14 3 34
    Peter Norvig 2012: 31 12 8
    Wolfgang Kessling 2012: 17 11 1
    Jon Nguyen 2012: 8 13 33
    Nirmalya Kumar 2012: 20 2 8
    E.O. Wilson 2012: 8 31 33
    Rives 2012: 19 5 13
    Massimo Banzi 2012: 25 26 1
    Jonathan Harris 2007: 14 1 25
    Don Tapscott 2012: 14 31 5
    Elyn Saks 2012: 2 1 3
    Boaz Almog 2012: 17 26 33
    Alanna Shaikh 2012: 16 0 1
    Raghava KK 2012: 5 2 0
    Cesar Harada 2012: 31 17 8
    Usman Riaz + Preston Reed 2012: 13 0 18
    Jane McGonigal 2012: 22 29 3
    Jonathan Eisen 2012: 1 27 26
    Chris Gerdes 2012: 9 8 17
    Larry Brilliant 2006: 14 28 1
    Emily Oster 2007: 8 30 7
    Marc Goodman 2012: 31 1 14
    Jared Ficklin 2012: 29 18 8
    Todd Humphreys 2012: 13 25 2
    Gabriel Barcia-Colombo 2012: 26 1 22
    Mina Bissell 2012: 27 32 2
    Jamie Drummond 2012: 8 14 1
    Baba Shiv 2012: 8 18 15
    Matt Mills 2012: 9 8 14
    Neil Harbisson 2012: 1 24 4
    John Graham-Cumming 2012: 31 13 8
    Will Wright 2007: 25 26 8
    Vinay Venkatraman 2012: 25 28 8
    James Stavridis 2012: 14 11 6
    Malte Spitz 2012: 10 2 15
    Tracy Chevalier 2012: 3 5 8
    Ramesh Raskar 2012: 33 8 10
    Michael Hansmeyer 2012: 31 25 34
    Noah Wilson-Rich 2012: 34 13 19
    Michael Anti 2012: 20 10 1
    Stephen Ritz 2012: 12 34 24
    Daphne Koller 2012: 12 31 1
    Rives 2007: 3 5 9
    Becci Manson 2012: 3 21 1
    Mark Applebaum 2012: 25 29 0
    Scilla Elworthy 2012: 1 2 9
    Margaret Heffernan 2012: 0 1 7
    Max Little 2012: 26 1 28
    Pam Warhurst 2012: 9 1 0
    Kirby Ferguson 2012: 18 31 34
    Mark Forsyth 2012: 1 13 15
    Lisa Kristine 2012: 1 23 6
    Ivan Krastev 2012: 8 1 20
    David Bolinsky 2007: 0 27 8
    Caitria + Morgan O'Neill 2012: 11 9 1
    Jon Ronson 2012: 2 9 5
    Timothy Prestero 2012: 15 14 25
    Rob Legato 2012: 25 26 8
    Robert Neuwirth 2012: 1 14 30
    Shyam Sankar 2012: 10 31 1
    Antony Gormley 2012: 33 13 8
    Jonathan Trent 2012: 21 8 17
    Kent Larson 2012: 1 34 0
    Scott Fraser 2012: 18 2 13
    Allison Hunt 2007: 5 34 26
    Vikram Patel 2012: 28 1 14
    Leslie T. Chang 2012: 0 2 16
    Susan Solomon 2012: 27 8 0
    Wayne McGregor 2012: 8 13 18
    Beth Noveck 2012: 10 23 11
    Tristram Stuart 2012: 24 14 1
    Sarah-Jayne Blakemore 2012: 4 0 1
    Julian Treasure 2012: 13 28 12
    Andrew Blum 2012: 14 5 34
    Bandi Mbubi 2012: 31 14 15
    George Ayittey 2007: 30 1 11
    Ed Gavagan 2012: 8 5 13
    Rachel Botsman 2012: 1 26 14
    Andrew McAfee 2012: 8 1 31
    Read Montague 2012: 1 4 25
    Clay Shirky 2012: 1 8 18
    Ben Goldacre 2012: 1 32 28
    Bahia Shehab 2012: 3 1 34
    Aris Venetikidis 2012: 34 25 4
    Vicki Arroyo 2012: 34 14 21
    Amy Cuddy 2012: 1 0 15
    Ngozi Okonjo-Iweala 2007: 30 1 8
    Robert Gupta 2012: 29 4 2
    Jason McCue 2012: 9 8 26
    Shimon Schocken 2012: 31 12 1
    Thomas P. Campbell 2012: 2 25 3
    Tim Leberecht 2012: 23 1 2
    John Maeda 2012: 25 31 5
    Ruby Wax 2012: 13 5 9
    Melissa Marshall 2012: 23 13 15
    Maurizio Seracini 2012: 13 23 2
    Eddie Obeng 2012: 9 14 16
    William Kamkwamba 2007: 17 7 9
    John Wilbanks 2012: 1 10 28
    Beau Lotto + Amy O'Toole 2012: 18 29 8
    Heather Brooke 2012: 13 10 9
    Ryan Merkley 2012: 9 31 10
    Pankaj Ghemawat 2012: 14 10 1
    David Pizarro 2012: 1 13 16
    Lemn Sissay 2012: 2 12 5
    Doris Kim Sung 2012: 26 25 23
    Marco Tempest 2012: 29 9 18
    Rory Stewart 2012: 1 11 20
    Euvin Naidoo 2007: 30 8 16
    Sanjay Pradhan 2012: 12 34 20
    Emma Teeling 2012: 8 0 19
    Adam Garone 2012: 32 2 8
    Faith Jegede 2012: 13 14 16
    Matt Killingsworth 2012: 1 0 7
    Jake Wood 2012: 34 12 9
    Gary Greenberg 2012: 25 14 27
    Georgette Mulheir 2012: 12 28 4
    Jeff Hancock 2012: 1 0 2
    Julie Burstein 2012: 2 3 23
    Patrick Awuah 2007: 30 12 13
    Arunachalam Muruganantham 2012: 8 6 11
    Hannah Brencher 2012: 3 13 11
    Leah Buechley 2012: 0 34 16
    David Binder 2012: 34 0 23
    Daphne Bavelier 2012: 29 26 0
    Amos Winter 2012: 0 1 31
    Sleepy Man Banjo Boys 2012: 29 8 3
    Louie Schwartzberg 2012: 13 3 22
    Candy Chang 2012: 22 34 15
    Ernesto Sirolli 2012: 1 30 2
    Nicholas Negroponte 2006: 0 8 12
    Chris Abani 2007: 5 30 0
    Jonas Eliasson 2012: 1 0 17
    Janine Shepherd 2012: 2 22 8
    Munir Virani 2012: 11 14 8
    Paolo Cardini 2012: 0 13 31
    Bobby Ghosh 2012: 1 11 14
    Ludwick Marishane 2012: 14 13 12
    Jeff Smith 2012: 22 8 15
    Nina Tandon 2012: 27 32 8
    Lemon Andersen 2012: 5 18 3
    Ellen 't Hoen 2012: 28 1 11
    Jacqueline Novogratz 2007: 2 30 0
    Markham Nolan 2012: 5 31 2
    Maz Jobrani 2012: 5 2 13
    Marcus Byrne 2012: 8 9 25
    Ben Saunders 2012: 22 5 19
    Robin Chase 2012: 0 1 8
    Molly Crockett 2012: 4 1 0
    Steven Addis 2012: 34 3 15
    Adam Davidson 2012: 1 8 0
    Ronny Edry 2012: 13 1 5
    Karen Thompson Walker 2013: 3 7 6
    Vusi Mahlasela 2007: 22 3 8
    Hadyn Parry 2013: 9 15 26
    Andy Puddicombe 2013: 22 13 3
    Don Levy 2013: 14 16 22
    Jonathan Haidt 2013: 9 26 0
    Sue Austin 2013: 34 1 3
    Jarrett J. Krosoczka 2013: 12 3 2
    Boghuma Kabisen Titanji 2013: 28 30 11
    Angela Patton 2013: 2 5 13
    Ellen Jorgensen 2013: 0 8 13
    Cameron Russell 2013: 1 15 13
    Jeff Skoll 2007: 1 0 14
    Richard Weller 2013: 32 28 1
    Colin Stokes 2013: 6 7 5
    Janine di Giovanni 2013: 1 22 3
    Colin Powell 2013: 12 9 5
    Steven Schwaitzberg 2013: 11 14 15
    Leslie Morgan Steiner 2013: 6 3 12
    Wingham Rowan 2013: 23 1 8
    Mitch Resnick 2013: 31 1 12
    iO Tillett Wright 2013: 1 13 3
    Fahad Al-Attiya 2013: 21 1 17
    Dean Kamen 2007: 2 8 9
    Zahra' Langhi 2013: 6 11 3
    Tyler DeWitt 2013: 27 13 5
    Cesar Kuriyama 2013: 13 0 3
    Lee Cronin 2013: 11 8 15
    Edi Rama 2013: 1 13 14
    Shabana Basij-Rasikh 2013: 12 6 3
    Erik Schlangen 2013: 21 26 17
    James B. Glattfelder 2013: 0 26 10
    Esther Perel 2013: 15 11 5
    Young-ha Kim 2013: 12 18 29
    Erin McKean 2007: 5 7 0
    Miguel Nicolelis 2013: 4 13 29
    Keith Chen 2013: 8 31 14
    Afra Raymond 2013: 8 20 2
    Andreas Schleicher 2013: 12 1 20
    Michael Dickinson 2013: 4 7 26
    Bruce Feiler 2013: 12 2 3
    Sugata Mitra 2013: 2 12 5
    Jennifer Granholm 2013: 2 8 17
    Amanda Palmer 2013: 1 0 25
    Andrew Mwenda 2007: 30 11 20
    Allan Savory 2013: 19 13 14
    Edith Widder 2013: 13 33 19
    Ron Finley 2013: 24 15 1
    Kakenya Ntaiya 2013: 12 6 2
    Shane Koyczan 2013: 2 12 3
    Dan Pallotta 2013: 17 1 32
    David Anderson 2013: 4 13 25
    Stewart Brand 2013: 27 23 8
    Bono 2013: 1 10 0
    Catarina Mota 2013: 16 23 31
    Theo Jansen 2007: 21 19 4
    Danny Hillis 2013: 26 0 1
    Elon Musk 2013: 17 7 9
    Hyeonseo Lee 2013: 3 22 20
    Francis Collins 2013: 27 32 11
    Eric Whitacre 2013: 21 3 13
    Jessica Green 2013: 0 34 8
    Mark Shaw 2013: 21 8 9
    Richard Turere 2013: 3 9 0
    Colin Camerer 2013: 4 1 8
    Kees Moeliker 2013: 3 5 1
    Steven Pinker 2007: 31 7 25
    Sanjay Dastoor 2013: 17 8 0
    Lawrence Lessig 2013: 1 20 2
    Skylar Tibbits 2013: 21 31 14
    Ken Jennings 2013: 5 8 25
    Freeman Hrabowski 2013: 12 2 23
    Keller Rinaudo 2013: 26 15 0
    Dan Ariely 2013: 1 2 7
    Eric Dishman 2013: 28 9 8
    Laura Snyder 2013: 6 31 3
    Rose George 2013: 21 9 8
    Steven Pinker 2007: 22 1 7
    Thomas Insel 2013: 4 7 28
    Joshua Prager 2013: 2 3 22
    Andres Lozano 2013: 4 8 28
    BLACK 2013: 3 22 14
    John McWhorter 2013: 1 7 31
    Robert Gordon 2013: 17 20 8
    Erik Brynjolfsson 2013: 31 13 5
    Jennifer Healey 2013: 8 5 10
    David Pogue 2013: 13 15 33
    Nilofer Merchant 2013: 16 8 28
    Jeff Han 2006: 0 25 13
    Deborah Scranton 2007: 5 0 8
    Taylor Wilson 2013: 0 17 21
    Sebastião Salgado 2013: 21 3 22
    Juan Enriquez 2013: 13 0 18
    Rita Pierson 2013: 2 12 16
    Timothy Bartik 2013: 20 1 12
    ShaoLan 2013: 20 6 31
    Bill Gates 2013: 12 11 0
    Ramsey Musallam 2013: 12 13 18
    Pearl Arredondo 2013: 12 8 11
    Malcolm London 2013: 11 12 1
    Zeresenay Alemseged 2007: 19 30 5
    Geoffrey Canada 2013: 5 12 18
    Angela Lee Duckworth 2013: 12 11 8
    Ken Robinson 2013: 12 1 5
    Meg Jay 2013: 5 3 23
    Maria Bezaitis 2013: 1 11 10
    Liu Bolin 2013: 1 23 20
    Jay Silver 2013: 13 25 14
    Sergey Brin 2013: 13 0 9
    Peter Singer 2013: 1 7 0
    Phil Hansen 2013: 13 3 26
    John Maeda 2007: 25 22 16
    Judy MacDonald Johnston 2013: 8 22 15
    Alastair Parvin 2013: 26 0 34
    Ji-Hae Park 2013: 29 22 13
    Paola Antonelli 2013: 0 5 29
    Jackson Katz 2013: 6 11 8
    Hendrik Poinar 2013: 26 19 30
    Lisa Bu 2013: 22 12 3
    Andrew Solomon 2013: 2 12 1
    Alex Laskey 2013: 17 1 2
    Anas Aremeyaw Anas 2013: 3 9 1
    Stephen Petranek 2007: 19 0 8
    Denise Herzing 2013: 21 31 0
    Martin Villeneuve 2013: 25 3 5
    Andrew McAfee 2013: 8 16 0
    Raffaello D'Andrea 2013: 31 21 33
    George Papandreou 2013: 11 20 1
    Daniel Suarez 2013: 10 11 31
    Manal al-Sharif 2013: 6 3 1
    Didier Sornette 2013: 13 20 17
    Juliana Rotich 2013: 30 9 31
    Joseph Kim 2013: 3 22 24
    Paul MacCready 2007: 13 19 8
    Paul Pholeros 2013: 23 9 28
    Camille Seaman 2013: 2 13 21
    Lesley Hazleton 2013: 22 7 3
    Peter Attia 2013: 5 1 13
    Bob Mankoff 2013: 34 8 0
    Michael Archer 2013: 8 7 27
    Rodney Brooks 2013: 8 1 31
    Eric X. Li 2013: 20 14 3
    Joel Selanikio 2013: 10 5 1
    Jinha Lee 2013: 14 31 33
    Carolyn Porco 2007: 19 33 21
    Sleepy Man Banjo Boys 2013: 29 8 5
    Charmian Gooch 2013: 17 0 1
    Michael Green 2013: 34 26 8
    Diana Reiss, Peter Gabriel, Neil Gershenfeld and Vint Cerf 2013: 31 8 9
    Jack Andraka 2013: 32 13 25
    Al Vernacchio 2013: 29 1 13
    Bernie Krause 2013: 19 9 3
    Gavin Pretor-Pinney 2013: 7 18 5
    Pico Iyer 2013: 0 9 3
    Miranda Wang and Jeanny Yao 2013: 21 7 26
    Tom Thum 2013: 13 18 5
    John Searle 2013: 4 7 5
    Kate Stone 2013: 25 13 16
    Roberto D'Angelo + Francesca Fedeli 2013: 0 22 13
    Paul Kemp-Robertson 2013: 26 7 1
    Tania Luna 2013: 3 13 19
    Bastian Schaefer 2013: 31 11 1
    Eli Beer 2013: 3 1 2
    Julie Taymor 2013: 3 5 8
    Peter van Manen 2013: 10 16 28
    Richard Branson 2007: 7 5 9
    Beardyman 2013: 8 16 0
    Daniel H. Cohen 2013: 7 15 9
    Jinsop Lee 2013: 5 18 33
    Saki Mafundikwa 2013: 30 18 1
    Eleanor Longden 2013: 3 2 1
    Derek Paravicini and Adam Ockelford 2013: 29 13 0
    Margaret Heffernan 2013: 1 2 5
    Shigeru Ban 2013: 1 34 2
    Russell Foster 2013: 4 11 8
    Steve Ramirez and Xu Liu 2013: 4 27 13
    Hod Lipson 2007: 26 25 16
    May El-Khalil 2013: 1 3 20
    Adam Spencer 2013: 5 9 2
    Kelly McGonigal 2013: 1 28 8
    Chrystia Freeland 2013: 20 7 11
    Alexa Meade 2013: 13 8 1
    George Monbiot 2013: 19 14 22
    Jake Barton 2013: 26 1 3
    Ron McCallum 2013: 31 1 3
    Sonia Shah 2013: 1 22 7
    Apollo Robbins 2013: 18 13 25
    Maira Kalman 2007: 5 0 2
    James Lyne 2013: 31 25 10
    Marla Spivak 2013: 24 11 14
    Eric Berlow and Sean Gourley 2013: 25 9 14
    Andras Forgacs 2013: 27 24 11
    Benjamin Barber 2013: 14 34 23
    Elizabeth Loftus 2013: 1 3 8
    Stuart Firestein 2013: 5 7 25
    Onora O'Neill 2013: 7 1 8
    James Flynn 2013: 2 1 14
    Kevin Breel 2013: 22 1 14
    Sirena Huang 2006: 26 29 2
    Jan Chipchase 2007: 1 16 25
    Malcolm Gladwell 2013: 9 18 3
    Kelli Swazey 2013: 22 1 28
    Amy Webb 2013: 10 8 0
    Fabian Oefner 2013: 13 9 18
    Jason Pontin 2013: 17 19 16
    Michael Porter 2013: 26 9 11
    Michael Sandel 2013: 22 12 1
    Janette Sadik-Khan 2013: 34 9 1
    Trita Parsi 2013: 20 3 11
    Gary Slutkin 2013: 34 28 0
    VS Ramachandran 2007: 4 2 18
    Andrew Fitzgerald 2013: 34 3 7
    Jeff Speck 2013: 34 17 9
    Amanda Bennett 2013: 3 2 32
    Iwan Baan 2013: 34 1 21
    Alessandro Acquisti 2013: 10 31 3
    Hetain Patel 2013: 21 20 2
    Steve Howard 2013: 9 17 1
    Charles Robertson 2013: 30 8 20
    Parul Sehgal 2013: 5 13 7
    Gian Giudice 2013: 33 3 13
    Eleni Gabre-Madhin 2007: 30 24 14
    Xavier Vilalta 2013: 34 33 0
    Mariana Mazzucato 2013: 26 0 20
    Mohamed Hijri 2013: 17 24 11
    Abha Dawesar 2013: 3 31 13
    Holly Morris 2013: 6 19 8
    Dong Woo Jang 2013: 3 12 34
    Rodrigo Canales 2013: 1 26 13
    Robin Nagle 2013: 23 34 3
    Grégoire Courtine 2013: 4 1 34
    Mikko Hypponen 2013: 5 10 20
    Sherwin Nuland 2007: 5 3 1
    Arthur Benjamin 2013: 18 33 31
    Dambisa Moyo 2013: 20 1 14
    Chris Downey 2013: 34 13 3
    Mohamed Ali 2013: 1 34 3
    Stefan Larsson 2013: 28 10 26
    Jane McGonigal 2013: 29 8 13
    Lian Pin Koh 2013: 31 9 13
    Greg Asner 2013: 8 19 0
    Henry Evans and Chad Jenkins 2013: 1 31 12
    Andreas Raptopoulos 2013: 14 34 31
    Matthieu Ricard 2007: 13 3 8
    Peter Doolittle 2013: 11 8 15
    Jared Diamond 2013: 1 28 12
    Suzana Herculano-Houzel 2013: 4 17 13
    David Steindl-Rast 2013: 1 14 22
    Toby Eccles 2013: 28 1 9
    Geraldine Hamilton 2013: 27 8 0
    Sally Kohn 2013: 0 13 9
    David Lang 2013: 9 3 13
    Enrique Peñalosa 2013: 34 20 17
    Boyd Varty 2013: 30 1 3
    Lawrence Lessig 2007: 31 1 11
    Diébédo Francis Kéré 2013: 12 23 21
    Eddy Cartaya 2013: 34 0 21
    Stephen Cave 2013: 22 13 3
    Rose George 2013: 19 8 5
    Toni Griffin 2013: 34 1 26
    Marco Annunziata 2013: 10 11 31
    Andrew Solomon 2013: 2 1 7
    Krista Donaldson 2013: 9 1 11
    Paul Piff 2013: 1 13 9
    Diana Nyad 2013: 2 8 9
    Paul Rothemund 2007: 25 31 18
    Mick Cornett 2014: 34 0 3
    Maysoon Zayid 2014: 3 9 2
    Suzanne Talhouk 2014: 2 15 31
    Roger Stein 2014: 32 26 1
    Sandra Aamodt 2014: 4 24 1
    Frederic Kaplan 2014: 26 11 19
    Ryan Holladay 2014: 0 34 29
    Harish Manwani 2014: 11 8 26
    Mark Kendall 2014: 9 31 14
    Sheryl Sandberg 2014: 6 2 0
    David Keith 2007: 7 11 0
    Luke Syson 2014: 25 26 3
    Guy Hoffman 2014: 25 13 29
    Shereen El Feki 2014: 6 22 14
    Paula Johnson 2014: 6 32 28
    Yves Morieux 2014: 11 34 1
    Joe Kowan 2014: 13 0 29
    Anant Agarwal 2014: 0 12 31
    Anne Milgram 2014: 10 1 16
    McKenna Pope 2014: 1 0 11
    Nicolas Perony 2014: 5 31 33
    Juan Enriquez 2007: 17 9 16
    Maya Penn 2014: 14 25 3
    Esta Soler 2014: 6 10 2
    Dan Berkenstock 2014: 10 13 16
    Teddy Cruz 2014: 34 0 20
    Alex Wissner-Gross 2014: 31 19 13
    Aparna Rao 2014: 25 23 0
    David Puttnam 2014: 28 11 23
    Leyla Acaroglu 2014: 11 16 26
    Chris McKnett 2014: 0 17 14
    Rupal Patel 2014: 31 8 1
    Larry Brilliant 2007: 14 19 30
    Yann Dall'Aglio 2014: 7 3 15
    Molly Stevens 2014: 0 27 26
    Roselinde Torres 2014: 1 14 9
    Christopher Ryan 2014: 6 18 14
    Ash Beckham 2014: 5 13 9
    Siddharthan Chandran 2014: 27 4 7
    Catherine Bracy 2014: 34 0 26
    Michael Metcalfe 2014: 30 17 19
    Henry Lin 2014: 33 11 7
    Siddharthan Chandran 2014: 27 4 7
    Jennifer Lin 2006: 8 29 15
    Robert Full 2007: 0 15 13
    Catherine Bracy 2014: 34 0 26
    Henry Lin 2014: 33 11 7
    Annette Heuser 2014: 20 0 18
    Mary Lou Jepsen 2014: 4 31 26
    Philip Evans 2014: 10 7 17
    Christopher Soghoian 2014: 31 20 0
    Gabe Barcia-Colombo 2014: 31 25 26
    Manu Prakash 2014: 26 8 13
    Ajit Narayanan 2014: 31 15 12
    Anne-Marie Slaughter 2014: 6 23 26
    Ron Eglash 2007: 2 30 13
    Toby Shapshak 2014: 30 0 14
    Carin Bondar 2014: 26 19 7
    Steven Pinker and Rebecca Newberger Goldstein 2014: 1 6 20
    Daniel Reisel 2014: 4 11 27
    Edward Snowden 2014: 9 11 7
    Chris Hadfield 2014: 33 13 16
    Charmian Gooch 2014: 0 11 14
    Richard Ledgett 2014: 7 1 16
    Larry Page 2014: 0 1 7
    Ziauddin Yousafzai 2014: 6 12 22
    Philippe Starck 2007: 5 16 15
    Bran Ferren 2014: 11 34 1
    Ed Yong 2014: 4 14 5
    Del Harvey 2014: 1 7 13
    Hugh Herr 2014: 31 18 14
    Geena Rocero 2014: 3 1 6
    TED staff 2014: 18 15 11
    Allan Adams 2014: 33 25 19
    Bill and Melinda Gates 2014: 0 9 8
    Jennifer Golbeck 2014: 1 10 16
    Lawrence Lessig 2014: 8 2 1
    Murray Gell-Mann 2007: 33 2 16
    Amanda Burden 2014: 34 1 21
    Christopher Emdin 2014: 12 18 9
    Louie Schwartzberg 2014: 16 10 3
    David Sengeh 2014: 2 6 31
    Gabby Giffords and Mark Kelly 2014: 3 0 9
    David Brooks 2014: 22 15 7
    Jennifer Senior 2014: 12 23 7
    Norman Spack 2014: 12 2 8
    Jeremy Kasdin 2014: 33 19 13
    Matthew Carter 2014: 2 31 10
    Amory Lovins 2007: 17 25 11
    Sarah Lewis 2014: 3 5 25
    Michel Laberge 2014: 17 25 0
    Hamish Jolly 2014: 21 31 26
    James Patten 2014: 31 29 1
    Elizabeth Gilbert 2014: 3 8 13
    Wendy Chung 2014: 8 26 4
    David Epstein 2014: 14 17 31
    Andrew Bastawrous 2014: 1 9 28
    Gavin Schmidt 2014: 19 8 10
    Sarah Jones 2014: 5 13 0
    Arthur Benjamin 2007: 18 5 8
    Stanley McChrystal 2014: 1 5 2
    Randall Munroe 2014: 10 5 1
    Mark Ronson 2014: 29 3 25
    William Black 2014: 11 17 13
    Deborah Gordon 2014: 21 32 10
    Kevin Briggs 2014: 13 3 22
    Tristram Wyatt 2014: 0 16 4
    Rives 2014: 13 3 25
    Simon Sinek 2014: 1 2 16
    Jackie Savitz 2014: 5 24 11
    Daniel Goleman 2007: 1 5 7
    Andrew Solomon 2014: 2 1 9
    Chris Kluwe 2014: 8 15 31
    Wes Moore 2014: 1 3 13
    Sebastian Junger 2014: 3 7 6
    Jon Mooallem 2014: 0 3 19
    Kitra Cahana 2014: 24 22 23
    Stephen Friend 2014: 8 26 10
    Sting 2014: 14 9 3
    Ray Kurzweil 2014: 4 9 19
    Dan Gilbert 2014: 1 19 7
    Lakshmi Pratury 2007: 3 13 15
    Stephen Burt 2014: 15 3 1
    Robert Full 2014: 31 26 15
    Yoruba Richen 2014: 3 20 13
    Stella Young 2014: 1 5 16
    Keren Elazari 2014: 14 7 16
    Will Potter 2014: 1 34 3
    Uri Alon 2014: 23 8 13
    AJ Jacobs 2014: 8 1 5
    Kwame Anthony Appiah 2014: 14 7 16
    Anne Curzan 2014: 31 8 7
    Gever Tulley 2007: 16 12 0
    Ruth Chang 2014: 14 7 22
    Jamila Lyiscott 2014: 15 7 1
    Billy Collins 2014: 3 5 24
    Shaka Senghor 2014: 6 22 16
    Lorrie Faith Cranor 2014: 1 26 0
    Naomi Oreskes 2014: 33 26 5
    Ge Wang 2014: 8 29 13
    Julian Treasure 2014: 1 0 14
    Chris Domas 2014: 0 10 31
    Sara Lewis 2014: 9 33 14
    Isabel Allende 2008: 6 14 15
    Simon Anholt 2014: 20 15 7
    Paul Bloom 2014: 1 15 7
    George Takei 2014: 20 3 1
    Joi Ito 2014: 8 11 16
    Nicholas Negroponte 2014: 8 2 1
    Renata Salecl 2014: 1 7 11
    Karima Bennoune 2014: 1 6 3
    David Kwong 2014: 8 18 15
    David Chalmers 2014: 4 7 33
    Nikolai Begg 2014: 8 18 2
    Amy Smith 2006: 11 21 16
    Yossi Vardi 2008: 13 11 28
    Shih Chieh Huang 2014: 16 33 1
    Heather Barnett 2014: 1 24 31
    Ze Frank 2014: 13 22 3
    Shai Reshef 2014: 12 11 30
    Margaret Gould Stewart 2014: 1 23 10
    Hubertus Knabe 2014: 1 0 20
    Janet Iwasa 2014: 10 0 27
    Megan Washington 2014: 5 9 3
    Talithia Williams 2014: 10 13 2
    Nick Hanauer 2014: 20 11 1
    Deborah Gordon 2008: 13 24 23
    Dan Pacholke 2014: 1 11 27
    Eric Liu 2014: 34 1 15
    Clint Smith 2014: 16 3 12
    Tim Berners-Lee 2014: 16 10 15
    Aziza Chaouni 2014: 21 34 26
    Jarrett J. Krosoczka 2014: 12 2 3
    Laurel Braitman 2014: 26 13 7
    Ziyah Gafić 2014: 19 20 3
    Martin Rees 2014: 19 13 33
    Rose Goslinga 2014: 30 11 2
    J.J. Abrams 2008: 5 25 16
    Meera Vijayann 2014: 6 3 20
    Sally Kohn 2014: 9 13 6
    Jill Shargaa 2014: 2 24 13
    Jim Holt 2014: 14 33 13
    Isabel Allende 2014: 15 3 4
    Shubhendu Sharma 2014: 25 17 11
    Colin Grant 2014: 3 12 22
    Zak Ebrahim 2014: 1 3 22
    Dan Barasch 2014: 34 0 13
    Hans and Ola Rosling 2014: 14 12 8
    David Gallo 2008: 13 21 18
    Uldus Bakhtiozina 2014: 6 34 12
    Rishi Manchanda 2014: 28 23 11
    Andrew Connolly 2014: 33 10 13
    Mac Barnett 2014: 13 12 5
    Avi Reichental 2014: 8 5 25
    Antonio Donato Nobre 2014: 21 5 11
    Lord Nicholas Stern 2014: 19 9 17
    Kenneth Cukier 2014: 10 8 31
    Eman Mohammed 2014: 6 3 23
    Matthew O'Reilly 2014: 28 22 11
    Paola Antonelli 2008: 25 0 23
    Moshe Safdie 2014: 34 22 9
    Francis de los Reyes 2014: 21 8 9
    Susan Colantuono 2014: 6 2 1
    Gail Reed 2014: 28 11 12
    Nancy Kanwisher 2014: 4 18 8
    Daria van den Bercken 2014: 29 0 3
    Thomas Piketty 2014: 20 10 8
    Meaghan Ramsey 2014: 12 1 7
    Pia Mancini 2014: 11 8 31
    Dilip Ratha 2014: 30 17 12
    Frank Gehry 2008: 8 1 23
    Glenn Greenwald 2014: 1 15 16
    Jeff Iliff 2014: 4 27 26
    Myriam Sidibe 2014: 28 12 7
    Jorge Soto 2014: 32 11 31
    Melissa Fleming 2014: 12 3 1
    Kitra Cahana 2014: 3 22 14
    Susan Etlinger 2014: 10 1 11
    Fred Swaniker 2014: 30 11 13
    Joy Sun 2014: 1 30 23
    Fabien Cousteau 2014: 13 11 3
    Raul Midon 2008: 9 18 5
    Marc Abrahams 2014: 1 9 34
    Kimberley Motley 2014: 1 11 3
    Sergei Lupashin 2014: 0 13 1
    Frans Lanting 2014: 22 1 3
    Debra Jarvis 2014: 32 8 13
    Jeremy Heimans 2014: 34 17 1
    Alessandra Orofino 2014: 34 1 11
    Ameenah Gurib-Fakim 2014: 30 5 24
    Kare Anderson 2014: 1 14 16
    Alejandro Aravena 2014: 34 18 17
    Bill Strickland 2008: 2 12 9
    Haas&Hahn 2014: 1 23 2
    Ramanan Laxminarayan 2014: 11 17 0
    Michael Green 2014: 20 14 8
    Ethan Nadelmann 2014: 1 28 20
    Leana Wen 2014: 28 5 2
    Vincent Moon and Naná Vasconcelos 2014: 8 0 1
    David Grady 2014: 1 13 18
    Will Marshall 2014: 19 13 10
    Nancy Frates 2014: 2 8 13
    Joe Landolina 2014: 26 27 23
    Ben Dunlap 2008: 2 3 5
    Rosie King 2014: 1 9 14
    Mark Plotkin 2014: 14 2 1
    Emily Balcetis 2014: 1 26 14
    Pico Iyer 2014: 8 3 22
    Oren Yakobovich 2014: 11 23 3
    Ben Saunders 2014: 9 24 3
    Rainer Strack 2014: 1 17 23
    Barbara Natterson-Horowitz 2014: 28 31 19
    Jose Miguel Sokoloff 2014: 0 2 8
    David Pogue 2008: 9 13 3
    Anastasia Taylor-Lind 2014: 6 3 33
    Thomas Hellum 2014: 1 3 25
    Catherine Crump 2014: 10 1 13
    Dave Troy 2014: 34 7 10
    Vernā Myers 2014: 5 8 6
    Jeremy Howard 2014: 31 16 26
    Carol Dweck 2014: 12 34 4
    Bruno Torturra 2014: 1 26 0
    Mundano 2014: 23 34 14
    Erin McKean 2014: 25 1 15
    Ross Lovegrove 2006: 16 5 21
    Alison Jackson 2008: 23 5 1
    Michael Rubinstein 2014: 31 13 26
    Asha de Vos 2015: 0 11 19
    Daniele Quercia 2015: 1 34 23
    Aziz Abu Sarah 2015: 1 0 24
    Fredy Peccerelli 2015: 1 6 8
    Tasso Azevedo 2015: 11 17 19
    Navi Radjou 2015: 17 21 20
    Robert Swan 2015: 19 9 1
    Robert Muggah 2015: 34 9 1
    Cristina Domenech 2015: 3 2 16
    Chris Anderson 2008: 8 25 7
    Matthieu Ricard 2015: 11 19 2
    Sarah Bergbreiter 2015: 0 11 16
    Joe Madiath 2015: 21 1 6
    Morgana Bailey 2015: 19 1 3
    Miguel Nicolelis 2015: 4 8 13
    Severine Autesserre 2015: 14 6 20
    Khadija Gbla 2015: 2 6 8
    Bassam Tariq 2015: 3 34 14
    Zeynep Tufekci 2015: 1 11 3
    Bruce Aylward 2015: 1 28 5
    Robin Chase 2008: 8 17 1
    Ben Ambridge 2015: 6 1 4
    Tom Wujec 2015: 0 1 25
    Brian Dettmer 2015: 7 0 16
    Jaap de Roode 2015: 7 28 19
    Ricardo Semler 2015: 1 16 8
    Kenneth Shinozuka 2015: 28 0 9
    Hannah Fry 2015: 1 7 13
    Guy Winch 2015: 28 3 13
    Nadine Burke Harris 2015: 28 4 32
    Laura Boushnak 2015: 6 12 8
    Jaime Lerner 2008: 34 23 15
    Angelo Vermeulen 2015: 0 26 1
    James A. White Sr. 2015: 2 13 11
    Rob Knight 2015: 13 28 0
    Khalida Brohi 2015: 6 8 3
    Romina Libster 2015: 1 28 34
    Ben Wellington 2015: 10 34 13
    Helder Guimarães 2015: 8 15 2
    Jon Gosier 2015: 14 31 16
    Topher White 2015: 1 26 3
    Harry Baker 2015: 1 2 3
    David Macaulay 2008: 25 13 0
    Andy Yen 2015: 26 0 10
    Ilona Szabó de Carvalho 2015: 14 11 1
    Sangu Delle 2015: 30 11 24
    Marc Kushner 2015: 34 5 19
    Ismael Nazario 2015: 1 2 8
    Shimpei Takahashi 2015: 10 34 7
    Linda Hill 2015: 1 5 23
    Vincent Cochetel 2015: 5 22 3
    Robyn Stein DeLuca 2015: 6 28 2
    David Eagleman 2015: 4 13 10
    Michael Pollan 2008: 25 0 14
    Joseph DeSimone 2015: 0 26 16
    Monica Lewinsky 2015: 1 3 13
    Fei-Fei Li 2015: 31 13 10
    Anand Giridharadas 2015: 22 3 14
    Dave Isay 2015: 1 3 22
    Theaster Gates 2015: 1 16 34
    Dame Stephanie Shirley 2015: 6 23 3
    Alison Killing 2015: 1 34 28
    Daniel Kish 2015: 1 8 13
    Kevin Rudd 2015: 20 14 3
    Howard Rheingold 2008: 1 31 34
    Boniface Mwangi 2015: 1 3 20
    Bill Gates 2015: 1 11 16
    Bel Pesce 2015: 1 22 0
    Eduardo Sáenz de Cabezón 2015: 18 2 5
    Dan Ariely 2015: 1 7 15
    Fred Jansen 2015: 26 33 19
    Barat Ali Batoor 2015: 3 1 21
    Kailash Satyarthi 2015: 12 1 3
    Takaharu Tezuka 2015: 12 5 2
    Paul Tudor Jones II 2015: 13 18 3
    Pamelia Kurstin 2008: 29 15 8
    Nathalie Cabrol 2015: 22 19 8
    Gary Haugen 2015: 13 14 26
    Jedidah Isler 2015: 33 10 3
    Chris Milk 2015: 1 14 3
    Clint Smith 2015: 14 12 21
    Nizar Ibrahim 2015: 21 19 0
    Nick Bostrom 2015: 7 31 0
    Greg Gage 2015: 8 4 18
    Sophie Scott 2015: 1 13 0
    Alice Goffman 2015: 1 12 3
    George Dyson 2008: 8 25 1
    Pamela Ronald 2015: 24 1 31
    Abe Davis 2015: 31 13 8
    Bill T. Jones 2015: 7 3 18
    Tal Danino 2015: 32 27 31
    Dawn Landes 2015: 3 18 13
    Anand Varma 2015: 27 0 3
    Elora Hardy 2015: 13 34 9
    Roman Mars 2015: 34 13 25
    The Lady Lifers 2015: 9 19 22
    Martine Rothblatt 2015: 1 13 2
    Moshe Safdie 2008: 2 34 25
    Cosmin Mihaiu 2015: 28 11 29
    Steven Wise 2015: 16 18 25
    Esther Perel 2015: 1 6 7
    Chris Burkard 2015: 13 21 3
    Jeffrey Brown 2015: 1 3 34
    Yassmin Abdel-Magied 2015: 5 9 14
    Sara Seager 2015: 33 22 13
    Jimmy Nelson 2015: 8 2 1
    Bill Gross 2015: 0 26 7
    Laura Schulz 2015: 8 10 13
    Richard Baraniuk 2006: 1 14 0
    Jill Sobule + Julia Sweeney 2008: 5 3 13
    Tony Fadell 2015: 13 3 16
    Trevor Aaronson 2015: 1 20 0
    Linda Cliatt-Wayman 2015: 12 8 5
    Suki Kim 2015: 14 5 3
    Sarah Jones 2015: 18 13 5
    Donald Hoffman 2015: 7 4 33
    Lee Mokobe 2015: 12 2 1
    Rana el Kaliouby 2015: 31 6 10
    Margaret Heffernan 2015: 1 0 11
    Steve Silberman 2015: 12 3 14
    Raspyni Brothers 2008: 18 8 9
    LaToya Ruby Frazier 2015: 6 28 1
    Roxane Gay 2015: 6 11 15
    Chip Kidd 2015: 0 7 13
    Maryn McKenna 2015: 1 3 28
    Chris Urmson 2015: 8 18 0
    Dame Ellen MacArthur 2015: 14 3 13
    Jimmy Carter 2015: 6 1 14
    Latif Nasser 2015: 3 28 22
    Gayle Tzemach Lemmon 2015: 6 2 3
    Joseph Lekuton 2008: 12 3 8
    Rajiv Maheswaran 2015: 5 16 29
    Memory Banda 2015: 6 3 2
    Ash Beckham 2015: 16 22 2
    Noy Thrupkaew 2015: 1 2 23
    Aspen Baker 2015: 6 3 22
    Alec Soth + Stacey Baker 2015: 2 3 26
    Salvatore Iaconesi 2015: 32 4 0
    Marlene Zuk 2015: 1 7 0
    Jon Ronson 2015: 1 3 8
    Alaa Murabit 2015: 6 12 3
    Steve Jurvetson 2008: 16 8 31
    John Green 2015: 14 1 34
    eL Seed 2015: 1 3 0
    Yuval Noah Harari 2015: 5 14 31
    Benedetta Berti 2015: 20 11 5
    Rich Benjamin 2015: 1 3 2
    Matt Kenyon 2015: 3 13 0
    Patience Mthunzi 2015: 27 28 31
    Alix Generous 2015: 1 16 3
    Manuel Lima 2015: 0 31 26
    Tony Wyss-Coray 2015: 4 27 19
    Roy Gould + Curtis Wong 2008: 33 8 7
    Christopher Soghoian 2015: 31 11 15
    Dustin Yellin 2015: 25 0 8
    Jim Al-Khalili 2015: 22 14 33
    Seth Berkley 2015: 1 26 11
    Robin Murphy 2015: 10 1 8
    Yves Morieux 2015: 1 11 17
    Wendy Freedman 2015: 33 9 8
    Elizabeth Nyamayaro 2015: 6 14 3
    Jamie Bartlett 2015: 8 14 1
    Jim Simons 2015: 23 2 3
    Alan Kay 2008: 25 8 12
    Alan Eustace 2015: 18 8 0
    Barry Schwartz 2015: 23 1 31
    BJ Miller 2015: 22 28 13
    Billie Jean King 2015: 6 0 13
    David Rothkopf 2015: 8 20 14
    Mia Birdsong 2015: 1 12 23
    Michael Kimmel 2015: 6 2 15
    Mandy Len Catron 2015: 3 0 15
    Scott Dinsmore 2015: 1 16 8
    Sakena Yacoobi 2015: 2 6 5
    Craig Venter 2008: 13 34 27
    Frances Larson 2015: 1 3 16
    Mary Robinson 2015: 14 11 1
    Robin Morgan 2015: 33 4 3
    Samuel Cohen 2015: 4 32 28
    Taiye Selasi 2015: 20 3 24
    Mac Stone 2015: 21 13 9
    Martin Pistorius 2015: 3 22 1
    Emilie Wapnick 2015: 15 16 12
    Alice Bows-Larkin 2015: 11 17 14
    Siddhartha Mukherjee 2015: 27 5 32
    Nicholas Negroponte 2008: 13 2 31
    Neri Oxman 2015: 34 31 22
    Teitur 2015: 13 2 9
    Vijay Kumar 2015: 31 21 26
    Alyson McGregor 2015: 6 28 13
    Anders Fjellberg 2015: 3 1 5
    Meklit Hadero 2015: 31 14 29
    Will Potter 2015: 2 1 7
    Jennifer Doudna 2015: 27 31 7
    Tom Uglow 2015: 26 7 31
    Francesco Sauro 2015: 19 0 14
    Jill Bolte Taylor 2008: 4 9 18
    Hilary Cottam 2015: 1 23 28
    Cesar Harada 2015: 21 12 10
    Christine Sun Kim 2015: 3 29 31
    Mathias Jud 2015: 1 14 3
    Daniel Levitin 2015: 16 4 8
    Nancy Lublin 2015: 10 15 1
    Melissa Fleming 2015: 2 21 1
    Patrícia Medici 2015: 11 5 14
    Harald Haas 2015: 27 17 33
    Frank Gehry 2008: 25 34 23
    Jenni Chang and Lisa Dazols 2015: 14 1 13
    Andreas Ekström 2015: 13 18 14
    Chelsea Shields 2015: 6 1 14
    Jean-Paul Mari 2015: 1 3 2
    Josh Luber 2015: 10 13 17
    Nonny de la Peña 2015: 0 1 25
    Anote Tong 2015: 7 9 1
    Carl Safina 2015: 5 4 16
    Genevieve von Petzinger 2015: 19 14 3
    Ann Morgan 2015: 14 3 1
    Majora Carter 2006: 34 1 11
    Jimmy Wales 2006: 1 8 0
    Dave Eggers 2008: 12 8 5
    Regina Hartley 2015: 1 12 9
    Marina Abramović 2015: 16 5 3
    Kristen Marhaver 2015: 23 19 9
    Jessica Shortall 2015: 23 6 3
    Chieko Asakawa 2015: 31 13 1
    Guillaume Néry 2015: 21 11 3
    Jedidah Isler 2015: 6 33 3
    Danit Peleg 2015: 13 3 26
    Raymond Wang 2015: 26 18 1
    Nicole Paris and Ed Cage 2015: 18 15 13
    Karen Armstrong 2008: 1 14 2
    Paul Greenberg 2015: 21 19 24
    Lucianne Walkowicz 2015: 19 33 22
    Alison Killing 2015: 1 25 15
    Jane Fonda and Lily Tomlin 2015: 6 5 11
    António Guterres 2015: 1 14 30
    Rodrigo Bijou 2015: 20 31 14
    Jason deCaires Taylor 2015: 0 23 7
    Robert Waldinger 2015: 1 22 28
    Harry Cliff 2016: 33 17 9
    Sebastian Wernicke 2016: 10 25 0
    Neil Turok 2008: 30 33 9
    Aomawa Shields 2016: 22 33 21
    David Sedlak 2016: 21 34 11
    James Veitch 2016: 2 8 9
    Tim Harford 2016: 11 26 7
    Melvin Russell 2016: 3 9 5
    Wael Ghonim 2016: 1 11 13
    Ole Scheeren 2016: 34 33 7
    Jill Farrant 2016: 21 24 8
    Oscar Schwartz 2016: 31 7 9
    Achenyo Idachaba 2016: 21 7 3
    Norman Foster 2008: 34 25 17
    Elizabeth Lev 2016: 3 14 33
    Yanis Varoufakis 2016: 20 14 23
    David Gruber 2016: 13 9 4
    Tania Simoncelli 2016: 0 32 19
    Auke Ijspeert 2016: 0 13 4
    Melati and Isabel Wijsen 2016: 12 11 3
    Linda Liukas 2016: 0 14 31
    Andrés Ruzo 2016: 21 13 9
    Judson Brewer 2016: 13 4 24
    Pardis Sabeti 2016: 23 10 13
    Christopher deCharms 2008: 4 8 31
    Matthew Williams 2016: 14 1 28
    Dambisa Moyo 2016: 20 7 14
    Sean Follmer 2016: 31 34 11
    Gregory Heyworth 2016: 14 31 3
    Mike Velings 2016: 11 24 19
    Dorothy Roberts 2016: 28 1 13
    Jocelyne Bloch 2016: 27 4 5
    Celeste Headlee 2016: 2 15 8
    Shonda Rhimes 2016: 29 23 3
    Allan Adams 2016: 33 16 0
    Clifford Stoll 2008: 8 5 16
    Raffaello D'Andrea 2016: 31 13 8
    Al Gore 2016: 17 19 8
    Dalia Mogahed 2016: 1 13 3
    Audrey Choi 2016: 11 17 26
    Mary Bassett 2016: 28 34 6
    Ivan Coyote 2016: 11 13 1
    Thomas Peschak 2016: 19 9 21
    Magda Sayeg 2016: 14 1 5
    Russ Altman 2016: 2 10 28
    Alexander Betts 2016: 11 14 15
    Siegfried Woldhek 2008: 18 5 3
    Travis Kalanick 2016: 1 8 9
    Reshma Saujani 2016: 6 8 18
    Caleb Harper 2016: 24 8 14
    Laura Robinson 2016: 19 8 9
    Mileha Soneji 2016: 11 0 14
    Tshering Tobgay 2016: 20 14 19
    Joe Gebbia 2016: 1 9 18
    Tim Urban 2016: 16 8 4
    Jessica Ladd 2016: 3 13 6
    Arthur Brooks 2016: 1 11 14
    David Hoffman 2008: 3 33 13
    Meron Gribetz 2016: 8 4 31
    Adam Foss 2016: 1 12 3
    Carol Fishman Cohen 2016: 23 1 3
    Latif Nasser 2016: 0 19 3
    Siyanda Mohutsiwa 2016: 30 0 3
    Alex Kipman 2016: 14 31 33
    Angélica Dass 2016: 1 23 12
    Dan Gross 2016: 1 13 12
    Lisa Nip 2016: 19 33 31
    Knut Haanaes 2016: 7 5 11
    Jakob Trollback 2008: 2 1 25
    Adam Grant 2016: 1 13 5
    Haley Van Dyck 2016: 23 11 20
    Parag Khanna 2016: 14 20 34
    Danielle Feinberg 2016: 21 14 33
    Tabetha Boyajian 2016: 10 33 13
    Robert Palmer 2016: 11 14 0
    Linus Torvalds 2016: 1 0 26
    Hugh Evans 2016: 14 11 3
    Stephen Petranek 2016: 19 21 8
    Paula Hammond 2016: 32 27 26
    Stephen Hawking 2008: 33 19 22
    Astro Teller 2016: 8 17 14
    Mary Norris 2016: 34 3 5
    Christiana Figueres 2016: 8 11 14
    Joshua Prager 2016: 22 3 19
    Chris Anderson 2016: 11 4 13
    Juan Enriquez 2016: 1 16 19
    Aditi Gupta 2016: 6 12 3
    Kenneth Lacovara 2016: 19 18 0
    Shivani Siroya 2016: 10 1 24
    R. Luke DuBois 2016: 9 10 0
    Mena Trott 2006: 1 5 16
    Al Gore 2008: 11 19 20
    Ameera Harouda 2016: 3 6 15
    Michael Metcalfe 2016: 17 30 11
    Riccardo Sabatini 2016: 31 22 8
    Sarah Gray 2016: 2 3 22
    Alice Rawsthorn 2016: 34 31 25
    Dan Pallotta 2016: 5 7 1
    Monica Byrne 2016: 2 13 22
    Jennifer Kahn 2016: 13 19 27
    Uri Hasson 2016: 4 8 1
    Sanford Biggers 2016: 23 14 33
    Johnny Lee 2008: 26 8 31
    Sangeeta Bhatia 2016: 32 8 25
    Kang Lee 2016: 12 28 31
    Moran Cerf 2016: 4 1 13
    Tod Machover + Dan Ellsey 2008: 29 25 0
    Yochai Benkler 2008: 1 31 16
    Ernest Madu 2008: 28 30 1
    Amy Tan 2008: 16 22 2
    Brian Greene 2008: 33 14 5
    Brian Cox 2008: 33 16 13
    They Might Be Giants 2008: 9 15 5
    Hector Ruiz 2008: 0 12 30
    Ze Frank 2006: 5 0 2
    Paul Stamets 2008: 19 22 21
    Paul Ewald 2008: 21 0 13
    Michael Moschen 2008: 8 18 25
    Joshua Klein 2008: 0 31 19
    Alisa Miller 2008: 14 5 20
    Mark Bittman 2008: 24 13 28
    Robert Ballard 2008: 8 19 22
    Yves Behar 2008: 0 23 34
    Arthur Ganson 2008: 25 13 8
    Seyi Oyesola 2008: 13 8 5
    Helen Fisher 2006: 6 4 7
    Paul Collier 2008: 9 20 14
    Susan Blackmore 2008: 7 16 31
    Nathan Myhrvold 2008: 5 25 0
    Wade Davis 2008: 19 14 3
    Murray Gell-Mann 2008: 19 16 7
    George Dyson 2008: 31 0 26
    Chris Jordan 2008: 13 25 1
    Dean Ornish 2008: 4 32 27
    Robert Full 2008: 8 13 23
    Adam Grosser 2008: 23 0 21
    Eve Ensler 2006: 6 2 0
    Steven Levitt 2008: 10 17 2
    Benjamin Zander 2008: 5 2 29
    Nicholas Negroponte 2008: 2 0 13
    Nellie McKay 2008: 13 32 18
    Peter Diamandis 2008: 2 33 5
    Rick Smolan 2008: 2 12 13
    Raul Midon 2008: 14 25 15
    Corneille Ewango 2008: 8 3 5
    Torsten Reil 2008: 8 13 26
    David Hoffman 2008: 2 9 13
    David Deutsch 2006: 33 19 16
    Clay Shirky 2008: 18 1 8
    Nellie McKay 2008: 13 22 15
    Freeman Dyson 2008: 22 19 33
    Helen Fisher 2008: 4 2 7
    Billy Graham 2008: 2 22 9
    AJ Jacobs 2008: 5 3 13
    Keith Barry 2008: 18 15 8
    Martin Seligman 2008: 22 1 15
    Marisa Fick-Jordan 2008: 34 30 23
    Chris Abani 2008: 2 6 0
    Richard Dawkins 2006: 14 33 13
    Louise Leakey 2008: 19 30 9
    Jonathan Harris 2008: 3 22 1
    Reed Kroloff 2008: 34 8 25
    Kevin Kelly 2008: 8 16 25
    Kwabena Boahen 2008: 4 10 8
    Robert Lang 2008: 25 8 33
    Bruno Bowden + Rufus Cappadocia 2008: 8 18 13
    Patricia Burchat 2008: 33 18 26
    Spencer Wells 2008: 19 30 14
    David Griffin 2008: 13 3 21
    Malcolm Gladwell 2006: 2 1 15
    Lennart Green 2008: 18 7 5
    Ian Dunbar 2008: 5 15 24
    Nellie McKay 2008: 13 2 11
    John Q. Walker 2008: 5 29 8
    Sugata Mitra 2008: 12 2 31
    Ory Okolloh 2008: 5 30 11
    Einstein the Parrot 2008: 0 18 15
    Paul Rothemund 2008: 25 31 8
    Peter Diamandis 2008: 8 5 33
    Peter Hirshberg 2008: 31 16 8
    Steven Levitt 2006: 0 1 8
    Jonathan Drori 2008: 18 1 9
    Jane Goodall 2008: 14 1 30
    Irwin Redlener 2008: 8 17 26
    Brewster Kahle 2008: 9 16 26
    David Gallo 2008: 21 8 19
    Carmen Agra Deedy 2008: 5 2 18
    Keith Bellows 2008: 9 2 8
    Ann Cooper 2008: 12 24 0
    Jonathan Haidt 2008: 1 7 0
    Eve Ensler 2008: 6 1 9
    Barry Schwartz 2006: 15 1 16
    David S. Rose 2008: 5 9 8
    Marvin Minsky 2008: 1 7 19
    Philip Zimbardo 2008: 1 8 2
    Laura Trice 2008: 11 15 0
    Caleb Chung 2008: 25 5 2
    Steven Pinker 2008: 1 4 12
    Rodney Brooks 2008: 8 5 15
    Stefan Sagmeister 2008: 26 34 2
    Noah Feldman 2008: 1 14 20
    Liz Diller 2008: 25 23 21
    Ken Robinson 2006: 2 12 7
    Dan Gilbert 2006: 18 15 0
    James Nachtwey 2008: 3 28 20
    David Perry 2008: 29 13 14
    Doris Kearns Goodwin 2008: 2 3 22
    Steven Johnson 2008: 25 5 1
    James Burchfield 2008: 13 18 3
    Garrett Lisi 2008: 33 34 5
    Paola Antonelli 2008: 0 23 5
    Virginia Postrel 2008: 14 5 7
    Dean Ornish 2008: 1 24 28
    John Hodgman 2008: 3 2 25
    Eva Vertes 2006: 32 27 0
    Paul MacCready 2008: 16 8 13
    Mihaly Csikszentmihalyi 2008: 1 22 23
    Kristen Ashburn 2008: 13 2 3
    Jared Diamond 2008: 11 16 20
    Rives 2008: 5 13 18
    Zach Kaplan + Keith Schacht 2008: 21 13 25
    Newton Aduaka 2008: 5 3 11
    Jackie Tabick 2008: 14 2 22
    Dayananda Saraswati 2008: 11 22 8
    James Forbes 2008: 2 28 3
    Aubrey de Grey 2006: 8 5 26
    Feisal Abdul Rauf 2008: 2 15 31
    Robert Thurman 2008: 7 2 15
    Robert Wright 2008: 1 7 14
    Graham Hawkes 2008: 13 21 8
    James Surowiecki 2008: 1 16 7
    John Francis 2008: 2 8 0
    Tim Brown 2008: 29 26 5
    Luca Turin 2008: 18 0 5
    Lee Smolin 2008: 33 16 8
    Samantha Power 2008: 1 14 20
    Iqbal Quadir 2006: 1 17 20
    Charles Elachi 2008: 5 19 25
    Ursus Wehrli 2008: 13 25 18
    Stewart Brand 2008: 18 19 8
    Isaac Mizrahi 2008: 5 18 7
    Franco Sacchi 2008: 0 5 30
    George Smoot 2008: 33 8 25
    Bill Joy 2008: 16 11 0
    Dan Barber 2008: 2 0 18
    Andy Hobsbawm 2008: 11 25 1
    Gregory Petsko 2008: 1 28 32
    Jacqueline Novogratz 2006: 1 30 6
    Richard Preston 2008: 19 13 3
    Philip Rosedale 2008: 22 5 8
    Larry Burns 2008: 17 5 14
    Nick Sears 2008: 26 34 13
    David Holt 2008: 13 2 5
    Eva Zeisel 2008: 23 16 26
    Dennis vanEngelsdorp 2008: 5 7 0
    Jay Walker 2008: 4 31 8
    Dan Gilbert 2008: 1 16 17
    Benjamin Wallace 2008: 5 14 2
    Ashraf Ghani 2006: 14 20 5
    Penelope Boston 2008: 22 19 7
    Steven Strogatz 2008: 18 13 8
    Nicholas Negroponte 2008: 12 14 19
    Jennifer 8. Lee 2008: 24 20 26
    Kary Mullis 2009: 5 25 8
    John Maeda 2009: 25 5 31
    Paul Sereno 2009: 0 9 8
    Paul Moller 2009: 8 9 17
    Greg Lynn 2009: 25 7 34
    Rob Forbes 2009: 25 34 0
    Sasa Vucinic 2006: 26 5 23
    Scott McCloud 2009: 13 25 1
    Peter Reinhart 2009: 22 25 24
    Joseph Pine 2009: 18 7 8
    Paula Scher 2009: 29 34 0
    David Carson 2009: 13 1 25
    Jamais Cascio 2009: 14 10 11
    Barry Schuler 2009: 5 13 31
    Sherwin Nuland 2009: 14 9 7
    Woody Norris 2009: 9 13 25
    Peter Ward 2009: 19 8 22
    Burt Rutan 2006: 8 25 33
    Aimee Mullins 2009: 5 13 9
    Joe DeRisi 2009: 5 13 25
    Natalie MacMaster 2009: 29 13 8
    Bill Gross 2009: 17 13 33
    Bill Gates 2009: 16 1 12
    Elizabeth Gilbert 2009: 5 23 8
    Milton Glaser 2009: 25 2 31
    David Merrill 2009: 31 29 25
    Barry Schwartz 2009: 1 11 18
    Juan Enriquez 2009: 9 8 19
    Ben Saunders 2006: 9 13 3
    Jose Antonio Abreu 2009: 12 29 14
    Gustavo Dudamel and the Teresa Carre√±o Youth Orchestra 2009: 14 29 25
    Sylvia Earle 2009: 19 21 22
    Jill Tarter 2009: 33 19 22
    Ed Ulbrich 2009: 31 3 10
    Charles Moore 2009: 19 21 24
    Richard Pyle 2009: 13 0 8
    Miru Kim 2009: 34 3 1
    Evan Williams 2009: 1 16 13
    Brenda Laurel 2009: 29 0 25
    Edward Burtynsky 2006: 16 8 23
    Willie Smits 2009: 1 21 19
    Nalini Nadkarni 2009: 1 9 0
    Mike Rowe 2009: 5 9 18
    Don Norman 2009: 21 0 25
    Pattie Maes + Pranav Mistry 2009: 0 31 10
    Aimee Mullins 2009: 12 1 2
    Stuart Brown 2009: 29 7 4
    Tim Berners-Lee 2009: 10 1 16
    Dan Dennett 2009: 7 13 4
    Dan Ariely 2009: 1 2 16
    Hans Rosling 2006: 30 10 14
    Robert Fischell 2006: 4 8 28
    Adam Savage 2009: 26 25 3
    Bruce McCall 2009: 8 5 25
    Kamal Meattle 2009: 17 11 14
    Saul Griffith 2009: 17 8 11
    Jacqueline Novogratz 2009: 2 0 1
    David Pogue 2009: 5 8 18
    John Wooden 2009: 16 3 2
    Nathan Wolfe 2009: 8 0 26
    C.K. Williams 2009: 3 6 7
    Jacek Utko 2009: 13 23 19
    Bono 2006: 30 14 0
    Ueli Gegenschatz 2009: 13 1 8
    Christopher C. Deam 2009: 2 16 34
    P.W. Singer 2009: 8 13 31
    Nathaniel Kahn 2009: 22 3 0
    Bruce Bueno de Mesquita 2009: 1 5 8
    Bonnie Bassler 2009: 27 13 9
    Emily Levine 2009: 5 2 7
    Renny Gleeson 2009: 18 11 8
    Shai Agassi 2009: 17 2 26
    Gregory Stock 2009: 8 0 13
    Michael Shermer 2006: 16 33 1
    JoAnn Kuchera-Morin 2009: 8 10 4
    Tim Ferriss 2009: 21 2 3
    Matthew Childs 2009: 8 9 26
    Margaret Wertheim 2009: 1 26 2
    Niels Diffrient 2009: 2 3 1
    Nate Silver 2009: 1 7 0
    Erik Hersman 2009: 30 26 10
    Ben Katchor 2009: 33 34 3
    Alex Tabarrok 2009: 14 20 30
    Michael Merzenich 2009: 4 26 8
    Peter Donnelly 2006: 2 9 5
    Sarah Jones 2009: 5 13 3
    Laurie Garrett 2009: 5 1 9
    Brian Cox 2009: 33 2 31
    Sean Gourley 2009: 10 8 5
    Mae Jemison 2009: 5 16 8
    Tom Shannon 2009: 25 13 33
    Al Gore 2009: 19 21 17
    Louise Fresco 2009: 24 11 26
    Seth Godin 2009: 1 11 15
    Hans Rosling 2009: 30 14 28
    Kevin Kelly 2006: 22 31 16
    Nandan Nilekani 2009: 20 1 0
    Naturally 7 2009: 3 23 2
    Ray Anderson 2009: 17 1 22
    Dan Ariely 2009: 1 2 15
    Mary Roach 2009: 2 6 8
    Carolyn Porco 2009: 21 19 22
    Yves Behar 2009: 0 9 17
    Joachim de Posada 2009: 12 15 19
    Jay Walker 2009: 15 14 22
    Michelle Obama 2009: 6 14 22
    Ray Kurzweil 2006: 31 26 19
    Jonathan Drori 2009: 16 14 19
    Kaki King 2009: 0 33 22
    Liz Coleman 2009: 14 12 20
    Ray Kurzweil 2009: 10 31 4
    Yann Arthus-Bertrand 2009: 22 19 5
    Felix Dennis 2009: 3 5 9
    Pete Alcorn 2009: 8 14 19
    Kevin Surace 2009: 9 1 5
    John La Grou 2009: 17 14 10
    Nancy Etcoff 2009: 0 1 4
    Peter Gabriel 2006: 14 3 0
    Robert Full 2009: 26 31 13
    Richard St. John 2009: 2 13 23
    Jane Poynter 2009: 8 24 22
    Clay Shirky 2009: 1 20 14
    Diane Benscoter 2009: 4 8 3
    Catherine Mohr 2009: 11 0 8
    Philip Zimbardo 2009: 3 8 1
    Paul Collier 2009: 11 20 1
    Katherine Fulton 2009: 34 8 16
    Ray Zahab 2009: 8 1 3
    Dean Ornish 2006: 1 32 28
    Arthur Benjamin 2009: 7 12 10
    Gever Tulley 2009: 12 16 25
    Daniel Libeskind 2009: 7 34 14
    Eames Demetrios 2009: 25 16 8
    Tom Wujec 2009: 4 25 33
    Sophal Ear 2009: 26 3 9
    Kary Mullis 2009: 16 2 0
    Stewart Brand 2009: 8 17 14
    Olafur Eliasson 2009: 33 34 21
    Daniel Kraft 2009: 27 9 13
    Rives 2006: 13 0 15
    Jim Fallon 2009: 4 1 9
    Nina Jablonski 2009: 1 19 13
    Gordon Brown 2009: 1 14 11
    Alain de Botton 2009: 7 1 15
    Golan Levin 2009: 1 25 0
    Elaine Morgan 2009: 9 2 21
    Willard Wigan 2009: 25 2 16
    Michael Pritchard 2009: 21 13 9
    Paul Romer 2009: 1 34 11
    Janine Benyus 2009: 21 26 25
    Richard St. John 2006: 9 2 0
    Emmanuel Jal 2009: 12 5 3
    Dan Pink 2009: 23 15 1
    Eric Giler 2009: 26 17 25
    Hans Rosling 2009: 14 30 20
    Natasha Tsakos 2009: 31 3 1
    Cary Fowler 2009: 11 14 15
    Joshua Silver 2009: 14 9 13
    Geoff Mulgan 2009: 1 7 16
    Evan Grant 2009: 26 33 10
    Steve Truglia 2009: 13 9 25
    Tony Robbins 2006: 5 1 9
    Robert Neuwirth 2007: 1 34 21
    James Balog 2009: 19 3 14
    Lewis Pugh 2009: 13 8 21
    Rebecca Saxe 2009: 4 1 8
    Misha Glenny 2009: 14 1 3
    Bjarke Ingels 2009: 26 25 21
    John Lloyd 2009: 2 5 16
    Oliver Sacks 2009: 2 4 1
    Imogen Heap 2009: 8 13 22
    Jonathan Zittrain 2009: 1 13 5
    Evgeny Morozov 2009: 26 1 7
    Bjorn Lomborg 2007: 14 16 1
    William Kamkwamba 2009: 12 2 21
    Taryn Simon 2009: 19 13 3
    Jacqueline Novogratz 2009: 1 11 23
    Parag Khanna 2009: 20 14 30
    Tim Brown 2009: 34 31 11
    Karen Armstrong 2009: 1 14 13
    Garik Israelian 2009: 26 33 19
    Stefan Sagmeister 2009: 0 26 3
    Carolyn Steel 2009: 24 34 8
    David Logan 2009: 1 22 5
    Phil Borges 2007: 1 12 13
    Chimamanda Ngozi Adichie 2009: 3 30 1
    Beau Lotto 2009: 18 8 33
    Sam Martin 2009: 33 0 34
    Eric Sanderson 2009: 34 11 21
    David Hanson 2009: 31 1 26
    Rory Sutherland 2009: 26 1 16
    Henry Markram 2009: 4 33 7
    Julian Treasure 2009: 13 29 1
    John Gerzema 2009: 26 0 9
    Paul Debevec 2009: 31 26 8
    Wade Davis 2007: 1 14 19
    Itay Talgam 2009: 5 29 18
    Marc Koska 2009: 1 21 12
    Ian Goldin 2009: 8 19 11
    David Deutsch 2009: 5 33 16
    Rachel Armstrong 2009: 34 9 26
    Becky  Blanton 2009: 3 1 0
    Marcus du Sautoy 2009: 9 25 31
    Stefana Broadbent 2009: 26 1 12
    Cameron Sinclair 2009: 9 17 24
    Rachel Pike 2009: 25 10 14
    Martin Rees 2007: 33 19 14
    Edward Burtynsky 2009: 17 19 14
    Cynthia Schneider 2009: 6 1 13
    Pranav Mistry 2009: 26 14 31
    Mathieu Lehanneur 2009: 25 4 23
    Fields Wicker-Miurin 2009: 14 1 5
    Devdutt Pattanaik 2009: 14 22 2
    Tom Wujec 2009: 5 31 13
    Hans Rosling 2009: 20 14 28
    Rob Hopkins 2009: 17 1 24
    Magnus Larsson 2009: 25 1 24
    Robert Wright 2007: 1 7 5
    Mallika Sarabhai 2009: 2 21 6
    Shashi Tharoor 2009: 20 14 5
    Gordon Brown 2009: 14 1 7
    Andrea Ghez 2009: 33 8 26
    Anupam Mishra 2009: 21 19 16
    Scott Kim 2009: 25 29 18
    Sunitha Krishnan 2009: 6 12 3
    Rory  Bremner 2009: 2 5 14
    Marc Pachter 2009: 2 22 5
    Thulasiraj Ravilla 2009: 28 7 0
    Steven Johnson 2007: 1 21 34
    Shereen El Feki 2009: 14 1 13
    Loretta Napoleoni 2009: 26 14 3
    Ryan Lobo 2009: 6 1 3
    Alexis Ohanian 2009: 26 13 0
    Charles Anderson 2009: 21 30 9
    James Geary 2009: 1 5 16
    Shaffi Mather 2009: 26 22 14
    Steven  Cowley 2009: 17 9 0
    Asher Hasan 2009: 0 13 28
    Bertrand Piccard 2010: 22 17 1
    Charles Leadbeater 2007: 9 1 7
    Vilayanur Ramachandran 2010: 4 19 31
    Nick Veasey 2010: 9 16 0
    Dan Buettner 2010: 22 19 1
    Romulus Whitaker 2010: 21 13 26
    Herbie Hancock 2010: 3 18 11
    Kartick Satyanarayan 2010: 12 1 20
    Kiran Sethi 2010: 12 34 2
    Lalitesh Katragadda 2010: 14 1 30
    Charles Fleischer 2010: 13 25 8
    David Blaine 2010: 3 0 21
    Anna Deavere Smith 2007: 5 2 8
    Ravin Agrawal 2010: 23 0 26
    Anthony Atala 2010: 27 26 13
    Bill Davenhall 2010: 0 22 18
    Joshua Prince-Ramus 2010: 26 8 1
    Eve Ensler 2010: 6 15 9
    Jane Chen 2010: 28 21 8
    Derek Sivers 2010: 13 5 15
    Sendhil Mullainathan 2010: 7 1 8
    Jamie Heywood 2010: 10 15 28
    George Whitesides 2010: 25 16 15
    Saul Griffith 2007: 16 25 5
    David Agus 2010: 32 9 10
    Tom Shannon, John Hockenberry 2010: 13 25 33
    Peter Eigen 2010: 14 20 30
    Jamie Oliver 2010: 24 12 5
    Blaise Agüera y Arcas 2010: 13 18 9
    David Cameron 2010: 1 7 8
    Aimee Mullins 2010: 22 12 1
    Bill Gates 2010: 17 11 16
    Kevin Kelly 2010: 31 22 16
    Philip K. Howard 2010: 1 12 9
    Joshua Prince-Ramus 2006: 34 26 8
    Neil Gershenfeld 2007: 31 13 9
    Eric Topol 2010: 28 0 16
    Temple Grandin 2010: 9 16 8
    Pawan Sinha 2010: 12 28 4
    Raghava KK 2010: 2 23 5
    Daniel Kahneman 2010: 7 22 1
    Harsha Bhogle 2010: 29 2 14
    Gary Flake 2010: 10 18 13
    James Cameron 2010: 5 0 8
    The LXD 2010: 8 13 25
    Srikumar Rao 2010: 22 15 5
    Carl Honoré 2007: 5 3 7
    Tim Berners-Lee 2010: 10 16 13
    Gary Lauder's new traffic sign 2010: 17 13 3
    Dan Barber 2010: 2 21 24
    Eric Mead 2010: 18 8 13
    Mark Roth 2010: 0 1 13
    Eric Dishman 2010: 28 8 10
    Jane McGonigal 2010: 29 14 22
    Ken Kamler 2010: 8 4 13
    Shekhar Kapur 2010: 2 3 5
    Sam Harris 2010: 7 6 14
    E.O. Wilson 2007: 19 22 14
    Juliana Machado Ferreira 2010: 1 24 19
    Alan Siegel 2010: 8 1 16
    Joel Levine 2010: 19 21 22
    Robert Gupta 2010: 29 8 25
    Kevin Bales 2010: 1 14 9
    Shukla Bose 2010: 12 14 8
    Kirk Citron 2010: 8 19 9
    Derek Sivers 2010: 0 9 1
    Adora Svitak 2010: 12 0 7
    Elizabeth Pisani 2010: 28 9 1
    James Nachtwey 2007: 3 1 30
    Dean Kamen 2010: 8 2 9
    Dennis Hong 2010: 0 13 31
    Jonathan Drori 2010: 0 26 13
    Natalie Merchant 2010: 9 3 15
    Michael Specter 2010: 5 1 15
    Jonathan Klein 2010: 14 1 7
    Catherine Mohr 2010: 17 21 16
    Thelma Golden 2010: 23 7 33
    Edith Widder 2010: 26 33 21
    James Randi 2010: 1 5 13
    Bill Clinton 2007: 28 14 1
    Frederick Balagadde 2010: 26 30 14
    Tom Wujec 2010: 18 0 23
    Omar Ahmad 2010: 26 8 9
    Kavita Ramdas 2010: 6 14 11
    Stephen Wolfram 2010: 33 16 31
    Roz Savage 2010: 13 8 22
    George Whitesides 2010: 16 7 5
    Sebastian Wernicke 2010: 8 15 10
    Esther Duflo 2010: 1 12 5
    Simon Sinek 2010: 1 13 4
    Chris Bangle 2007: 5 8 0
    Jeremy Jackson 2010: 0 21 1
    Anil Gupta 2010: 1 11 20
    Thomas Dolby 2010: 9 3 34
    Nicholas Christakis 2010: 1 16 25
    Nathan Myhrvold 2010: 26 0 25
    Enric Sala 2010: 21 1 0
    Dan Meyer 2010: 0 12 18
    Julia Sweeney 2010: 2 5 13
    William Li 2010: 32 24 26
    Graham Hill 2010: 28 3 21
    Craig Venter 2007: 13 19 27
    Dee Boersma 2010: 0 8 5
    Richard Sears 2010: 17 13 8
    Craig Venter 2010: 27 26 22
    Ken Robinson 2010: 2 1 7
    Johanna Blakley 2010: 0 26 15
    Sharmeen Obaid-Chinoy 2010: 12 15 28
    Seth Berkley 2010: 27 0 5
    Lawrence Lessig 2010: 18 25 1
    John Underkoffler 2010: 14 31 25
    Brian Skerry 2010: 19 21 5
    Dean Kamen 2007: 1 17 9
    Christopher "moot" Poole" 2010: 9 1 13
    Brian Cox 2010: 33 13 19
    Adam Sadowsky 2010: 16 25 0
    Michael Sandel 2010: 18 2 7
    John Kasaona 2010: 30 16 3
    Rory Sutherland 2010: 26 16 14
    Stewart Brand + Mark Z. Jacobson 2010: 17 14 8
    David Byrne 2010: 25 29 1
    Michael Shermer 2010: 13 18 4
    Margaret Gould Stewart 2010: 13 31 34
    Jane Goodall 2007: 14 5 31
    Peter Tyack 2010: 31 13 19
    Cameron Herold 2010: 12 2 13
    Ananda Shankar Jayant 2010: 32 22 2
    Chip Conley 2010: 26 3 22
    Marian Bantjes 2010: 23 0 13
    Charles Leadbeater 2010: 12 11 1
    Aditi Shankardass 2010: 4 12 26
    Hillel Cooperman 2010: 13 2 1
    Clay Shirky 2010: 8 2 31
    Ellen Dunham-Jones 2010: 0 34 13
    Golan Levin 2007: 13 7 31
    Stephen Palumbi 2010: 1 19 24
    Carter Emmart 2010: 33 19 26
    Mitchell Joachim 2010: 25 27 9
    Benoit Mandelbrot 2010: 16 3 2
    Ellen  Gustafson 2010: 24 0 14
    Nalini Nadkarni 2010: 7 6 19
    Hans Rosling 2010: 14 12 20
    Carl Safina 2010: 0 21 1
    Matt Ridley 2010: 1 19 25
    Ethan Zuckerman 2010: 14 26 1
    Julia Sweeney 2006: 2 13 0
    Janine Benyus 2007: 21 22 2
    Elif Shafak 2010: 3 1 5
    Julian Assange 2010: 5 1 13
    Naif Al-Mutawa 2010: 2 14 34
    Dimitar Sasselov 2010: 22 33 19
    Tan Le 2010: 4 31 26
    Kevin Stone 2010: 8 13 32
    Sheena Iyengar 2010: 2 1 11
    John Delaney 2010: 19 34 8
    Laurie Santos 2010: 26 0 8
    Lewis Pugh 2010: 21 13 9
    Seth Godin 2007: 1 15 8
    Jason Clay 2010: 23 11 8
    Sheryl WuDunn 2010: 12 6 5
    Peter Molyneux 2010: 13 9 8
    Jamil Abu-Wardeh 2010: 3 11 13
    Maz Jobrani 2010: 2 8 15
    Seth Priebatsch 2010: 1 29 0
    David McCandless 2010: 10 25 13
    Lee Hotz 2010: 19 5 23
    Jim Toomey 2010: 25 0 9
    Lisa Margonelli 2010: 17 26 11
    Thom Mayne 2007: 34 25 23
    Dan Cobley 2010: 33 14 7
    Nic Marks 2010: 22 11 7
    Johan Rockstrom 2010: 19 11 20
    His Holiness the Karmapa 2010: 7 14 3
    Derek Sivers 2010: 23 11 3
    Rachel Sussman 2010: 19 26 16
    Sugata Mitra 2010: 2 12 31
    Alwar Balasubramaniam 2010: 13 5 2
    Carne Ross 2010: 16 26 3
    Ben Cameron 2010: 23 31 5
    Vik Muniz 2007: 5 3 26
    Rob Dunbar 2010: 19 5 21
    Chris Anderson 2010: 14 1 31
    Jessa Gamble 2010: 8 3 1
    Nicholas Christakis 2010: 1 10 31
    Caroline Phillips 2010: 29 25 19
    Christien Meindertsma 2010: 26 0 1
    Steven Johnson 2010: 25 1 9
    Mitchell Besser 2010: 28 30 6
    Annie Lennox 2010: 30 5 28
    Fabian Hemmert 2010: 18 15 31
    James Watson 2007: 5 2 0
    Julian Treasure 2010: 29 13 28
    Gary Wolf 2010: 13 10 1
    Sebastian Seung 2010: 4 5 13
    Inge Missmahl 2010: 28 1 15
    Mechai Viravaidya 2010: 12 2 1
    Eben Bayer 2010: 26 17 8
    Tim Jackson 2010: 13 26 1
    Barbara Block 2010: 10 26 19
    Hans Rosling 2010: 30 28 12
    Stacey Kramer 2010: 22 34 5
    Frans Lanting 2007: 22 21 19
    Stefano Mancuso 2010: 4 13 25
    Melinda Gates 2010: 1 28 10
    Peter Haas 2010: 34 18 8
    Natalie Jeremijenko 2010: 21 28 26
    Ze Frank 2010: 1 13 2
    Jessica Jackley 2010: 3 7 16
    Heribert Watzke 2010: 4 24 26
    Dianna Cohen 2010: 21 26 9
    Patrick Chappatte 2010: 14 5 30
    David Byrne, Ethel + Thomas Dolby 2010: 9 2 25
    Paul Bennett 2007: 16 0 25
    R.A. Mashelkar 2010: 2 17 1
    Joseph Nye 2010: 20 17 8
    Barton Seaver 2010: 24 11 16
    Shimon Steinberg 2010: 13 26 18
    Miwa Matreyek 2010: 22 33 3
    Tom Chatfield 2010: 1 29 0
    David Bismark 2010: 1 16 25
    Greg Stone 2010: 2 21 19
    Gero Miesenboeck 2010: 4 27 33
    Andrew Bird 2010: 14 9 8
    Nick Bostrom 2007: 13 1 7
    Emily Pilloton 2010: 12 34 23
    Stefan Wolff 2010: 11 20 30
    Aaron Huey 2010: 20 1 6
    Auret van Heerden 2010: 20 11 15
    Eric Berlow 2010: 15 24 26
    Conrad Wolfram 2010: 7 1 14
    Denis Dutton 2010: 19 31 5
    Shimon Schocken 2010: 12 3 22
    John Hardy 2010: 12 5 14
    Kristina Gjerde 2010: 11 26 19
    Stefan Sagmeister 2007: 5 26 13
    Kim Gorgens 2010: 12 5 29
    Zainab Salbi 2010: 6 12 3
    Jason Fried 2010: 23 1 16
    Dan Phillips 2010: 13 25 2
    Birke Baehr 2010: 24 5 15
    William Ury 2010: 3 1 30
    Marcel Dicke 2010: 24 13 14
    Bart Weetjens 2010: 24 26 1
    Arthur Potts Dawson 2010: 9 21 24
    Halla Tomasdottir 2010: 6 16 1
    Alex Steffen 2007: 5 1 16
    Tony Porter 2010: 6 2 13
    Kiran Bedi 2010: 2 12 3
    Hanna Rosin 2010: 6 13 8
    Diana Laufenberg 2010: 12 31 0
    Rufus Griscom + Alisa Volkman 2010: 0 3 7
    Rachel Botsman 2010: 15 26 8
    Beverly + Dereck Joubert 2010: 8 3 0
    Sheryl Sandberg 2010: 6 0 15
    Majora Carter 2010: 1 0 13
    Brené Brown 2010: 2 1 5
    Rick Warren 2006: 2 22 8
    Susan Savage-Rumbaugh 2007: 31 7 29
    Barry Schwartz 2010: 1 11 18
    Arianna Huffington 2011: 6 8 25
    Lesley Hazleton 2011: 6 3 1
    Charles Limb 2011: 4 13 8
    Deborah Rhodes 2011: 6 32 31
    Neil Pasricha 2011: 3 9 13
    Jody Williams 2011: 11 14 6
    Amber Case 2011: 31 2 16
    Thomas Thwaites 2011: 25 2 21
    Elizabeth Lesser 2011: 1 5 18
    Sheila Patek 2007: 26 13 0
    Ali Carr-Chellman 2011: 12 11 6
    Naomi Klein 2011: 13 17 11
    Charity Tillemann-Dick 2011: 3 8 2
    Van Jones 2011: 1 8 9
    Anders Ynnerman 2011: 10 0 4
    Heather Knight 2011: 10 8 31
    Martin Jacques 2011: 20 14 7
    Thomas Goetz 2011: 1 28 26
    Liza Donnelly 2011: 6 5 7
    Ariel Garten 2011: 31 5 0
    Al Seckel 2007: 8 25 16
    Bruce Feiler 2011: 2 3 9
    Kate Orff 2011: 34 21 0
    Dale Dougherty 2011: 16 8 13
    Johanna Blakley 2011: 8 6 1
    Christopher McDougall 2011: 19 9 16
    Suheir Hammad 2011: 6 8 11
    Nigel Marsh 2011: 22 23 11
    Cynthia Breazeal 2011: 0 31 1
    Hawa Abdi + Deqo Mohamed 2011: 1 6 28
    Michael Pawlyn 2011: 21 13 11
    Juan Enriquez 2007: 22 0 27
    Geert Chatrou 2011: 26 14 8
    Krista Tippett 2011: 33 3 7
    Patricia Kuhl 2011: 4 8 31
    Jacqueline Novogratz 2011: 6 2 1
    Lisa Gansky 2011: 16 0 17
    Madeleine Albright 2011: 6 7 2
    Noreena Hertz 2011: 11 6 23
    Iain Hutchison 2011: 1 32 9
    Elizabeth Lindsey 2011: 14 6 8
    Danny Hillis 2011: 32 0 8
    Nora York 2007: 15 11 13
    Ahn Trio 2011: 29 14 13
    Wadah Khanfar 2011: 1 14 3
    JR 2011: 13 14 6
    Wael Ghonim 2011: 1 8 26
    Bill Gates 2011: 11 16 0
    Anthony Atala 2011: 26 27 8
    Courtney Martin 2011: 6 0 26
    Salman Khan 2011: 12 13 18
    Deb Roy 2011: 21 10 3
    Rob Harmon 2011: 21 18 1
    Jill Sobule 2007: 2 3 18
    David Brooks 2011: 1 4 0
    Janna Levin 2011: 33 8 19
    Mark Bezos 2011: 22 11 8
    Rogier van der Heide 2011: 33 0 13
    Sarah Kay 2011: 5 16 3
    Hans Rosling 2011: 1 17 14
    Isabel Behncke 2011: 29 14 13
    Paul Root Wolpe 2011: 31 27 4
    Eythor Bender 2011: 19 31 26
    Claron McFadden 2011: 25 2 13
    Caroline Lavelle 2007: 2 9 33
    Patricia Ryan 2011: 12 5 15
    Ralph Langner 2011: 26 10 15
    Handspring Puppet Co. 2011: 25 8 3
    Sebastian Thrun 2011: 1 22 17
    Eric Whitacre 2011: 1 13 3
    AnnMarie Thomas 2011: 29 0 12
    Stanley McChrystal 2011: 9 2 3
    Chade-Meng Tan 2011: 14 15 1
    Morgan Spurlock 2011: 8 3 1
    Mick Ebeling 2011: 8 2 9
    Dan Dennett 2007: 13 8 1
    Caroline Casey 2011: 5 8 3
    Jackson Browne 2011: 14 8 15
    David Christian 2011: 33 19 13
    Dave Meslin 2011: 1 15 26
    Roger Ebert 2011: 31 3 1
    Marcin Jakubowski 2011: 9 14 13
    Susan Lim 2011: 27 22 14
    Sam Richards 2011: 5 15 20
    Kathryn Schulz 2011: 18 7 5
    John Hunter 2011: 2 12 9
    Evelyn Glennie 2007: 29 13 16
    Anil Ananthaswamy 2011: 33 21 23
    Ric Elias 2011: 22 9 16
    Harvey Fineberg 2011: 19 27 22
    Bruce Schneier 2011: 1 11 15
    Angela Belcher 2011: 27 17 0
    Mike Matas 2011: 13 17 15
    Arvind Gupta 2011: 13 25 12
    Eli Pariser 2011: 11 25 0
    Aicha el-Wafi + Phyllis Rodriguez 2011: 5 6 3
    Carlo Ratti 2011: 26 21 34
    William McDonough 2007: 34 17 14
    Suzanne Lee 2011: 11 21 26
    Sean Carroll 2011: 33 5 13
    Louie Schwartzberg 2011: 22 3 14
    Paul Nicklen 2011: 9 21 13
    Fiorenzo Omenetto 2011: 21 16 26
    Ron Gutman 2011: 26 4 22
    Amit Sood 2011: 8 0 26
    Leonard Susskind 2011: 2 7 0
    Ed Boyden 2011: 4 27 8
    Thomas Heatherwick 2011: 13 34 9
    Dan Dennett 2006: 13 15 7
    Jeff Bezos 2007: 0 5 3
    Elliot Krane 2011: 27 4 3
    Edith Widder 2011: 13 33 8
    Terry Moore 2011: 22 2 5
    Aaron Koblin 2011: 1 0 26
    Bruce Aylward 2011: 12 1 14
    Shirin Neshat 2011: 6 20 1
    Mustafa Akyol 2011: 20 6 26
    Dennis Hong 2011: 31 8 10
    Stefan Sagmeister 2011: 0 25 13
    Rives 2007: 8 18 13
    Aaron O'Connell 2011: 0 13 29
    Jessi Arrington 2011: 8 1 18
    Damon Horowitz 2011: 13 18 10
    Jack Horner 2011: 26 25 0
    Janet Echelman 2011: 34 3 23
    Paul Romer 2011: 34 9 2
    Alice Dreger 2011: 1 0 15
    JD Schramm 2011: 1 22 3
    Daniel Kraft 2011: 28 8 27
    Shea Hembrey 2011: 8 23 25
    Eddi Reader 2007: 9 31 13
    Steve Keil 2011: 29 11 8
    Camille Seaman 2011: 21 19 22
    Onyx Ashanti 2011: 25 29 34
    Maya Beiser 2011: 29 3 31
    Bill Ford 2011: 8 0 11
    Daniel Tammet 2011: 1 14 33
    Jok Church 2011: 5 22 9
    Honor Harger 2011: 33 19 31
    Joshua Walters 2011: 13 5 7
    Emiliano Salinas 2011: 11 8 22
    Eddi Reader 2007: 7 13 25
    Rajesh Rao 2011: 31 1 5
    Dave deBronkart 2011: 28 2 10
    Robert Hammond 2011: 34 1 8
    Matt Cutts 2011: 3 22 16
    Nathan Myhrvold 2011: 25 24 1
    Jonathan Drori 2011: 0 5 25
    Simon Lewis 2011: 4 3 28
    Nina Tandon 2011: 27 0 24
    Rebecca MacKinnon 2011: 20 11 31
    Maajid Nawaz 2011: 20 11 1
    Tom Honey 2007: 16 2 15
    Tim Harford 2011: 14 2 23
    Nadia Al-Sakkaf 2011: 6 8 1
    Mikko Hypponen 2011: 26 13 31
    Thandie Newton 2011: 22 3 0
    Kevin Slavin 2011: 2 13 26
    Markus Fischer 2011: 17 25 5
    Rory Stewart 2011: 11 1 20
    Geoffrey West 2011: 8 34 22
    Paul Bloom 2011: 1 15 23
    Josette Sheeran 2011: 24 12 1
    Richard Dawkins 2007: 1 15 2
    Julian Treasure 2011: 14 13 3
    Adam Ostrow 2011: 8 7 31
    Harald Haas 2011: 10 33 17
    Mark Pagel 2011: 31 14 0
    Jessica Green 2011: 10 28 33
    Philip Zimbardo 2011: 6 5 15
    Eve Ensler 2011: 32 6 3
    Alex Steffen 2011: 0 11 17
    Dyan deNapoli 2011: 13 3 17
    Jeremy Gilley 2011: 2 9 8
    Tom Rielly 2007: 8 25 13
    Lucianne Walkowicz 2011: 33 0 22
    Marco Tempest 2011: 2 6 1
    Dan Ariely 2011: 2 10 18
    Svante Pääbo 2011: 30 1 19
    Julia Bacha 2011: 8 20 3
    Skylar Tibbits 2011: 34 11 0
    Joan Halifax 2011: 6 14 0
    Edward Tenner 2011: 0 31 16
    Sarah Kaminsky 2011: 3 22 12
    Lee Cronin 2011: 22 8 11
    Rachelle Garniez 2007: 7 17 9
    Raghava KK 2011: 12 5 25
    Yasheng Huang 2011: 20 6 10
    Misha Glenny 2011: 1 31 23
    Kate Hartman 2011: 26 0 13
    Richard Resnick 2011: 32 19 22
    Lauren Zalaznick 2011: 15 19 3
    Niall Ferguson 2011: 20 7 0
    Jean-Baptiste Michel + Erez Lieberman Aiden 2011: 1 10 0
    Amy Lockwood 2011: 1 13 7
    Elizabeth Murchison 2011: 32 27 1
    Chris Anderson 2007: 31 8 17
    Sunni Brown 2011: 1 7 11
    Abraham Verghese 2011: 28 2 3
    Geoff Mulgan 2011: 12 16 23
    Jarreth Merz 2011: 1 30 3
    Ben Goldacre 2011: 10 32 5
    Danielle de Niese 2011: 5 33 15
    Yang Lan 2011: 1 20 8
    Christoph Adami 2011: 22 26 8
    Graham Hill 2011: 9 8 25
    Mike Biddle 2011: 14 25 17
    Natalie MacMaster 2007: 21 34 19
    Charles Hazlewood 2011: 29 8 0
    Alison Gopnik 2011: 26 13 12
    Richard Seymour 2011: 16 5 8
    Ian Ritchie 2011: 14 31 2
    Pamela Meyer 2011: 8 5 13
    Jae Rhim Lee 2011: 24 32 13
    Bunker Roy 2011: 2 6 12
    Justin Hall-Tipping 2011: 8 17 14
    Guy-Philippe Goldstein 2011: 14 5 13
    Todd Kuiken 2011: 13 25 0

```python

# Fit the LDA model
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)
print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
```

    
    Topics in LDA model:
    
    Topic 0:
    people 0.17, going 0.12, know 0.12, talk 0.11, car 0.11, computer 0.11, way 0.1, thing 0.1, actually 0.1, able 0.1, world 0.1, different 0.09, film 0.09, device 0.09, did 0.09, use 0.09, think 0.09, really 0.09, got 0.09, power 0.09, 
    
    Topic 1:
    world 0.1, violence 0.1, cells 0.09, patients 0.09, cancer 0.09, war 0.09, father 0.08, day 0.08, today 0.08, just 0.08, make 0.08, work 0.08, field 0.08, time 0.08, cell 0.08, stand 0.08, use 0.08, life 0.08, people 0.07, state 0.07, 
    
    Topic 2:
    years 1272.46, energy 1044.08, planet 952.69, earth 921.22, world 814.61, water 762.98, ve 719.5, ocean 691.31, going 642.01, life 609.82, oil 572.08, just 563.98, time 556.73, need 533.48, climate 509.74, sea 507.03, fish 482.79, species 470.4, ice 454.26, year 430.16, 
    
    Topic 3:
    light 1121.88, just 922.57, space 896.63, universe 869.77, time 728.21, look 625.82, different 567.79, way 542.17, world 490.07, science 486.24, actually 428.66, new 421.71, image 364.43, make 362.95, life 351.52, right 343.26, inside 334.95, ve 326.83, use 322.39, stars 320.79, 
    
    Topic 4:
    natural 0.08, oil 0.08, gas 0.07, million 0.07, dollars 0.07, use 0.07, carbon 0.06, ca 0.06, going 0.06, don 0.06, make 0.06, product 0.06, cost 0.06, products 0.06, energy 0.06, countries 0.06, plan 0.06, machine 0.06, look 0.06, point 0.06, 
    
    Topic 5:
    women 1980.24, men 1086.24, woman 577.72, compassion 391.21, girls 386.65, story 349.32, stories 324.73, sex 322.42, people 312.94, book 259.3, man 259.26, violence 255.47, world 250.97, said 242.0, books 240.23, mother 212.84, work 208.16, love 202.44, word 201.24, girl 188.86, 
    
    Topic 6:
    just 0.14, ve 0.13, said 0.13, years 0.13, year 0.13, going 0.13, earth 0.12, think 0.12, need 0.11, time 0.11, people 0.11, problem 0.1, let 0.1, way 0.1, percent 0.1, end 0.09, come 0.09, change 0.09, energy 0.09, say 0.09, 
    
    Topic 7:
    magic 229.91, audience 19.55, technology 5.61, try 5.15, digital 5.1, come 4.91, reality 4.88, hey 4.7, let 4.52, share 4.47, today 3.45, practice 3.07, tools 3.02, thank 2.79, right 2.65, research 2.63, form 2.41, oh 2.4, going 2.25, tell 2.06, 
    
    Topic 8:
    world 0.41, cancer 0.39, people 0.38, know 0.32, make 0.29, ve 0.28, actually 0.28, just 0.25, need 0.24, really 0.24, ll 0.24, don 0.2, book 0.2, today 0.2, going 0.2, work 0.19, technology 0.19, things 0.19, little 0.19, drug 0.19, 
    
    Topic 9:
    computer 742.0, machine 660.76, computers 335.41, machines 205.54, car 106.28, software 105.38, going 91.3, race 82.46, program 79.35, built 77.9, learning 77.84, technology 72.9, device 61.48, able 57.6, little 55.27, just 54.22, human 50.11, ve 49.41, thing 48.34, did 47.46, 
    
    Topic 10:
    just 0.6, people 0.53, going 0.5, world 0.41, really 0.38, think 0.38, actually 0.36, years 0.27, ve 0.24, new 0.24, make 0.23, need 0.23, carbon 0.22, time 0.22, food 0.21, let 0.21, systems 0.21, kind 0.21, things 0.21, know 0.2, 
    
    Topic 11:
    robot 456.37, robots 354.81, legs 141.13, want 97.21, fly 82.28, control 66.55, build 64.54, video 51.56, foot 47.96, just 45.8, body 43.45, actually 43.45, use 42.11, ll 41.89, second 41.15, doing 40.49, machines 40.38, human 36.61, look 36.23, really 35.87, 
    
    Topic 12:
    people 2888.28, think 2416.03, actually 1744.33, just 1708.03, things 1597.93, world 1540.33, going 1523.06, really 1510.64, want 1249.61, way 1220.45, know 1205.6, don 1152.73, ve 1141.0, right 1018.53, thing 980.37, kind 961.07, time 910.32, say 869.94, make 835.44, use 782.02, 
    
    Topic 13:
    animals 0.11, water 0.1, just 0.08, pretty 0.08, going 0.08, time 0.08, did 0.08, look 0.08, really 0.08, community 0.08, lot 0.08, right 0.08, life 0.08, feet 0.08, way 0.07, went 0.07, creating 0.07, fish 0.07, ve 0.07, surface 0.07, 
    
    Topic 14:
    going 0.43, people 0.42, think 0.42, don 0.32, just 0.3, said 0.3, really 0.29, ve 0.29, know 0.28, years 0.28, cells 0.27, time 0.27, world 0.25, make 0.24, little 0.24, look 0.24, things 0.23, life 0.22, got 0.2, say 0.2, 
    
    Topic 15:
    really 0.14, data 0.14, images 0.12, make 0.11, actually 0.11, time 0.1, just 0.1, use 0.1, work 0.09, science 0.09, numbers 0.09, art 0.09, single 0.09, information 0.09, way 0.09, come 0.09, lot 0.09, women 0.09, ve 0.08, read 0.08, 
    
    Topic 16:
    world 0.27, going 0.22, used 0.2, know 0.2, years 0.19, ve 0.18, let 0.18, really 0.18, use 0.18, people 0.17, look 0.16, time 0.16, think 0.16, percent 0.16, points 0.15, dollars 0.15, thing 0.15, make 0.15, great 0.15, new 0.15, 
    
    Topic 17:
    people 0.32, world 0.3, ve 0.27, really 0.23, book 0.19, time 0.19, just 0.18, way 0.17, years 0.17, story 0.17, film 0.17, going 0.17, things 0.16, work 0.15, images 0.14, life 0.14, make 0.14, didn 0.14, kind 0.14, tell 0.14, 
    
    Topic 18:
    use 0.36, water 0.32, need 0.27, don 0.23, think 0.19, right 0.18, going 0.18, make 0.18, people 0.17, percent 0.16, green 0.16, plant 0.15, world 0.15, ve 0.15, species 0.15, just 0.15, lot 0.14, using 0.14, way 0.14, years 0.14, 
    
    Topic 19:
    music 808.47, play 554.31, sound 489.05, listening 216.65, hear 183.47, sounds 163.44, playing 139.65, listen 129.3, video 94.73, way 47.8, experience 42.65, just 36.42, let 36.06, heard 32.47, really 26.42, going 23.74, ll 20.67, kind 19.42, song 18.53, little 16.6, 
    
    Topic 20:
    world 0.23, people 0.15, percent 0.15, better 0.14, today 0.14, violence 0.13, father 0.12, technology 0.11, stand 0.11, human 0.11, make 0.11, want 0.11, day 0.1, hope 0.1, time 0.1, use 0.1, good 0.1, information 0.1, mobile 0.1, tell 0.1, 
    
    Topic 21:
    people 0.31, women 0.26, know 0.2, thing 0.2, say 0.16, country 0.15, africa 0.15, think 0.14, business 0.13, just 0.13, world 0.13, ve 0.13, dollars 0.13, years 0.12, want 0.12, school 0.12, things 0.12, wonderful 0.12, love 0.11, universe 0.11, 
    
    Topic 22:
    cancer 955.22, design 852.35, just 776.43, work 763.14, really 733.42, people 682.11, actually 668.48, need 619.6, way 608.49, patients 603.25, new 602.5, patient 544.61, time 540.39, students 526.62, make 525.33, things 517.93, ve 517.55, health 511.94, think 458.63, know 455.06, 
    
    Topic 23:
    ca 592.18, ok 454.43, yeah 356.2, mean 169.0, chris 150.45, think 126.5, yes 111.33, right 93.86, ted 74.8, audience 71.22, good 61.94, got 61.49, thank 56.98, ve 48.74, did 41.64, ll 39.34, don 37.61, people 37.2, going 36.31, seven 30.8, 
    
    Topic 24:
    food 1175.17, water 1056.43, eat 348.86, plants 343.73, plant 320.99, bacteria 308.97, ve 241.59, world 241.5, use 217.96, just 217.24, waste 215.43, need 192.32, think 184.19, grow 173.29, feed 164.3, environment 159.73, species 158.72, make 158.26, natural 153.9, way 151.46, 
    
    Topic 25:
    brain 1002.72, self 328.02, sleep 259.93, mind 234.76, happiness 223.93, mental 182.88, body 154.36, going 109.26, think 84.67, minds 73.98, course 72.62, brains 68.2, memory 66.02, map 65.29, don 60.86, experience 60.01, region 53.06, day 50.13, life 49.57, night 48.86, 
    
    Topic 26:
    people 3466.36, percent 1425.01, going 1173.96, ve 1109.82, data 1095.29, just 1039.23, think 1014.68, dollars 929.89, money 888.45, need 847.79, know 795.48, years 786.16, make 772.66, world 759.18, don 754.35, time 744.18, want 738.47, things 709.81, actually 701.35, business 688.05, 
    
    Topic 27:
    world 0.09, actually 0.08, powerful 0.08, got 0.07, power 0.07, just 0.07, look 0.07, different 0.07, used 0.07, year 0.07, country 0.07, case 0.07, don 0.07, time 0.07, century 0.07, space 0.07, great 0.07, way 0.06, think 0.06, bring 0.06, 
    
    Topic 28:
    black 414.65, song 266.48, love 60.36, baby 31.33, white 23.89, cause 21.0, hear 16.52, sun 16.39, hey 15.82, ll 15.04, ve 12.84, going 10.04, oh 8.65, police 8.35, men 8.21, life 7.77, young 7.61, just 7.02, older 6.96, blood 6.38, 
    
    Topic 29:
    just 2288.76, going 2273.23, really 2129.24, actually 2126.5, think 1731.66, know 1584.15, little 1546.23, things 1321.03, look 1267.08, ve 1253.93, right 1188.04, brain 1176.42, different 1126.5, way 1101.53, cells 1044.21, thing 1043.15, don 969.8, make 945.46, kind 933.66, human 923.46, 
    
    Topic 30:
    cells 0.24, environment 0.17, place 0.16, feeling 0.16, help 0.15, people 0.14, women 0.13, brain 0.13, change 0.12, wall 0.12, cell 0.12, car 0.11, knows 0.11, pattern 0.11, remember 0.11, think 0.1, memory 0.09, said 0.09, believe 0.09, animal 0.09, 
    
    Topic 31:
    world 2095.75, people 1330.37, country 1229.41, countries 1019.62, children 822.07, india 789.48, years 788.44, china 737.65, today 674.62, states 632.53, war 583.6, global 583.42, government 574.55, united 557.72, percent 513.03, change 498.14, political 494.93, state 452.06, economic 446.6, chinese 422.42, 
    
    Topic 32:
    people 1287.07, city 1251.01, just 911.09, new 887.53, building 825.09, really 824.86, going 730.16, cities 723.54, ve 617.79, make 616.21, think 603.35, car 569.9, way 542.11, time 506.1, work 503.15, little 480.37, kind 475.19, design 466.69, years 459.25, things 446.35, 
    
    Topic 33:
    know 5554.6, people 5222.97, just 4589.86, said 4511.61, going 3605.3, don 3447.42, time 3176.95, ve 2831.94, think 2740.11, got 2712.46, life 2542.43, really 2533.77, want 2518.02, say 2359.3, did 2324.05, way 2134.6, years 2090.14, day 2045.77, things 2035.45, right 1982.55, 
    
    Topic 34:
    brain 0.41, going 0.24, look 0.18, mind 0.17, need 0.17, climate 0.16, think 0.16, just 0.16, earth 0.16, energy 0.16, fact 0.15, course 0.15, make 0.14, self 0.14, don 0.13, years 0.13, people 0.13, really 0.13, body 0.13, know 0.13,



```python
# Now to associate topics to documents...
doc_topic_distrib = lda.transform(tf)
```


```python
print(type(doc_topic_distrib), len(doc_topic_distrib))
```

    <class 'numpy.ndarray'> 2106

```python

# For a quick check of the number of documents and terms in the matrix:
print(dtm.shape)
```

```python

doctopic_orig = doctopic.copy()
num_groups = len(set(authordate))
doctopic_grouped = np.zeros((num_groups, n_topics))

for i, name in enumerate(sorted(set(authordate))):
    doctopic_grouped[i, :] = np.mean(doctopic[authordate == name, :], axis=0)
```

    /opt/local/Library/Frameworks/Python.framework/Versions/3.4/lib/python3.4/site-packages/ipykernel/__main__.py:6: VisibleDeprecationWarning: using a boolean instead of an integer will result in an error in the future
