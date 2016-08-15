

```python

```

## Some Useful Sample Scripts

The discussion and scripts below are from [Sujit Pal](http://sujitpal.blogspot.com/2014/08/topic-modeling-with-gensim-over-past.html). I'm copying and pasting them here because his website is a little unreadable.

In the code below, I read the text of each file, pass the words through gensim's tokenizer and filter out stopwords (from NLTK's English stopword list) using our custom MyCorpus class. These words are used to create a dictionary and BoW corpus, which is serialized to files for use in the next step.


```python
# Source: bow_model.py
import logging
import os
import nltk
import gensim

def iter_docs(topdir, stoplist):
    for fn in os.listdir(topdir):
        fin = open(os.path.join(topdir, fn), 'rb')
        text = fin.read()
        fin.close()
        yield (x for x in 
            gensim.utils.tokenize(text, lowercase=True, deacc=True, 
                                  errors="ignore")
            if x not in stoplist)

class MyCorpus(object):

    def __init__(self, topdir, stoplist):
        self.topdir = topdir
        self.stoplist = stoplist
        self.dictionary = gensim.corpora.Dictionary(iter_docs(topdir, stoplist))
        
    def __iter__(self):
        for tokens in iter_docs(self.topdir, self.stoplist):
            yield self.dictionary.doc2bow(tokens)


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)

TEXTS_DIR = "/path/to/texts/dir"
MODELS_DIR = "/path/to/models/dir"

stoplist = set(nltk.corpus.stopwords.words("english"))
corpus = MyCorpus(TEXTS_DIR, stoplist)

corpus.dictionary.save(os.path.join(MODELS_DIR, "mtsamples.dict"))
gensim.corpora.MmCorpus.serialize(os.path.join(MODELS_DIR, "mtsamples.mm"), 
                                  corpus)
```

I didn't know (and didn't have an opinion about) how many topics this corpus should yield so I decided to compute this by reducing the features to two dimensions, then clustering the points for different values of K (number of clusters) to find an optimum value. Gensim offers various transforms that allow us to project the vectors in a corpus to a different coordinate space. One such transform is the Latent Semantic Indexing (LSI) transform, which we use to project the original data to 2D.


```python
# Source: lsi_model.py
import logging
import os
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)

MODELS_DIR = "/path/to/models/dir"

dictionary = gensim.corpora.Dictionary.load(os.path.join(MODELS_DIR, 
                                            "mtsamples.dict"))
corpus = gensim.corpora.MmCorpus(os.path.join(MODELS_DIR, "mtsamples.mm"))

tfidf = gensim.models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]

# project to 2 dimensions for visualization
lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)

# write out coordinates to file
fcoords = open(os.path.join(MODELS_DIR, "coords.csv"), 'wb')
for vector in lsi[corpus]:
    if len(vector) != 2:
        continue
    fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
fcoords.close()
```

Next I clustered the points in the reduced 2D LSI space using KMeans, varying the number of clusters (K) from 1 to 10. The objective function used is the Inertia of the cluster, defined as the sum of squared differences of each point to its cluster centroid. This value is provided directly from the [Scikit-Learn KMeans algorithm][skk]. Other popular functions include Distortion (Inertia divided by the number of points) or the Percentage of Variance Explained, as described on this [StackOverflow post][so].

[skk]: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
[so]: http://stackoverflow.com/questions/6645895/calculating-the-percentage-of-variance-measure-for-k-means


```python
# Source: num_topics.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

MODELS_DIR = "/path/to/models/dir"
MAX_K = 10

X = np.loadtxt(os.path.join(MODELS_DIR, "coords.csv"), delimiter="\t")
ks = range(1, MAX_K + 1)

inertias = np.zeros(MAX_K)
diff = np.zeros(MAX_K)
diff2 = np.zeros(MAX_K)
diff3 = np.zeros(MAX_K)
for k in ks:
    kmeans = KMeans(k).fit(X)
    inertias[k - 1] = kmeans.inertia_
    # first difference    
    if k > 1:
        diff[k - 1] = inertias[k - 1] - inertias[k - 2]
    # second difference
    if k > 2:
        diff2[k - 1] = diff[k - 1] - diff[k - 2]
    # third difference
    if k > 3:
        diff3[k - 1] = diff2[k - 1] - diff2[k - 2]

elbow = np.argmin(diff3[3:]) + 3

plt.plot(ks, inertias, "b*-")
plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
plt.ylabel("Inertia")
plt.xlabel("K")
plt.show()
```

I plotted the Inertias for different values of K, then used Vincent Granville's approach of calculating the third differential to find an [elbow point][ep]. The elbow point happens here for K=5 and is marked with a red dot in the graph below.

[ep]: http://radimrehurek.com/2014/03/data-streaming-in-python-generators-iterators-iterables/

I then re-ran the KMeans algorithm with K=5 and generated the clusters.


```python
# Source: viz_topics_scatter.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

MODELS_DIR = "/path/to/models/dir"
NUM_TOPICS = 5

X = np.loadtxt(os.path.join(MODELS_DIR, "coords.csv"), delimiter="\t")
kmeans = KMeans(NUM_TOPICS).fit(X)
y = kmeans.labels_

colors = ["b", "g", "r", "m", "c"]
for i in range(X.shape[0]):
    plt.scatter(X[i][0], X[i][1], c=colors[y[i]], s=10)    
plt.show()
```

I then ran the full [LDA transform][ldat] against the BoW corpus, with the number of topics set to 5. As in LSI, I load up the corpus and dictionary from files, then apply the transform to project the documents into the LDA Topic space. Notice that LDA and LSI are conceptually similar in gensim - both are transforms that map one vector space to another.

[ldat]: http://radimrehurek.com/gensim/models/ldamodel.html


```python
# Source: lda_model.py
import logging
import os
import gensim

MODELS_DIR = "/path/to/models/dir"
NUM_TOPICS = 5

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)

dictionary = gensim.corpora.Dictionary.load(os.path.join(MODELS_DIR, 
                                            "mtsamples.dict"))
corpus = gensim.corpora.MmCorpus(os.path.join(MODELS_DIR, "mtsamples.mm"))

# Project to LDA space
lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS)
lda.print_topics(NUM_TOPICS)
```

The results are as follows. As you can see, each topic is made up of a mixture of terms. The top 10 terms from each topic is shown in the output below and covers between 69-80% of the probability space for each topic.


```python
import os
import wordcloud

MODELS_DIR = "models"

final_topics = open(os.path.join(MODELS_DIR, "final_topics.txt"), 'rb')
curr_topic = 0
for line in final_topics:
    line = line.strip()[line.rindex(":") + 2:]
    scores = [float(x.split("*")[0]) for x in line.split(" + ")]
    words = [x.split("*")[1] for x in line.split(" + ")]
    freqs = []
    for word, score in zip(words, scores):
        freqs.append((word, score))
    elements = wordcloud.fit_words(freqs, width=120, height=120)
    wordcloud.draw(elements, "gs_topic_%d.png" % (curr_topic),
                   width=120, height=120)
    curr_topic += 1
final_topics.close()
```
