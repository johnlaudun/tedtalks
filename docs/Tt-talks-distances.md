
# TEDtalk Similarities/Distances

## Preliminaries


```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib



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
```


```python
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


```python
import numpy as np  # a conventional alias
import sklearn.feature_extraction.text as text

vectorizer = text.CountVectorizer(input=talks, stop_words='english', min_df=15)

# These next two steps go straight to arrays, skipping sparse matrix and list.
dtm = vectorizer.fit_transform(talks).toarray() 
vocab = np.array(vectorizer.get_feature_names())

print(len(vocab), dtm.shape)
```

    8882 (2106, 8882)



```python
from sklearn.metrics.pairwise import cosine_similarity

dist = 1 - cosine_similarity(dtm)
np.round(dist, 2)
```




    array([[ 0.  ,  0.56,  0.58, ...,  0.65,  0.8 ,  0.71],
           [ 0.56,  0.  ,  0.57, ...,  0.7 ,  0.75,  0.67],
           [ 0.58,  0.57,  0.  , ...,  0.64,  0.73,  0.75],
           ..., 
           [ 0.65,  0.7 ,  0.64, ...,  0.  ,  0.79,  0.79],
           [ 0.8 ,  0.75,  0.73, ...,  0.79,  0.  ,  0.8 ],
           [ 0.71,  0.67,  0.75, ...,  0.79,  0.8 , -0.  ]])




```python
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

for x, y, citation in zip(xs, ys, authordate):
    plt.scatter(x, y)
    plt.text(x, y, citation)
    
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-13-d58d6cf2b32a> in <module>()
          5 pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
          6 
    ----> 7 for x, y, citation in zip(xs, ys, authordate):
          8     plt.scatter(x, y)
          9     plt.text(x, y, citation)


    NameError: name 'xs' is not defined

