```python
>>> # TEDtalks: Topics with LDA
...
... # =-=-=-=-=-=
... # Read CSV into DataFrame and then create lists
... # =-=-=-=-=-=
...
... import pandas
>>> import re
...
>>> # Create pandas dataframe
... colnames = ['author', 'title', 'date' , 'length', 'text']
>>> df = pandas.read_csv('../data/talks-v1b.csv', names=colnames)
...
>>> # Get all the texts in a list
... talks = df.text.tolist()
...
>>> # Getting only the years from dates list
... authors = df.author.tolist()
>>> dates = df.date.tolist()
>>> years = [re.sub('[A-Za-z ]', '', item) for item in dates]
...
>>> # Combining year with presenter for citation
... authordate = [author+" "+year for author, year in zip(authors, years)]
...
>>> # Just to check to see if things are synced,
... # let's create a new df with the two lists.
...
... cited_texts = pandas.DataFrame(
...     {'citation': authordate,
...      'text': talks,
...     })
```

```python
>>> cited_texts.head()
                        citation  \
0                   Al Gore 2006
1               David Pogue 2006
2          Cameron Sinclair 2006
3  Sergey Brin + Larry Page 2007
4          Nathalie Miebach 2011

                                                text
0  Thank you so much  Chris. And it's truly a gre...
1  Hello voice mail  my old friend. I've called f...
2  I'm going to take you on a journey very quickl...
3  Sergey Brin  I want to discuss a question I kn...
4  What you just heard are the interactions of ba...
```

```python
>>> # =-=-=-=-=-=
... # Remove the "empty" talks from both lists.
... # =-=-=-=-=-=
...
... # First, establish which talks are empty
... i = 0
>>> no_good = []
>>> for talk in talks:
...     A = type(talk)
...     B = type('string or something')
...     if A != B:
...         no_good.append(i)
...     i = i + 1
...
>>> # Second, delete in reverse order so as to preserve index order
... for index in sorted(no_good, reverse=True):
...     del talks[index]
>>> for index in sorted(no_good, reverse=True):
...     del authordate[index]
```

```python
>>> print(no_good)
[185, 398, 513, 877, 1015, 1100, 2011]
```

```python
>>> # =-=-=-=-=-=
... # LDA Topic Models
... # =-=-=-=-=-=
...
... # Documentation: https://pypi.python.org/pypi/lda
... # LDA requires a DTM as input
...
... import sklearn.feature_extraction.text as text
>>> import numpy as np
>>> import lda
...
>>> # Doc for DTM: https://de.dariah.eu/tatom/topic_model_python.html
... vectorizer = text.CountVectorizer(input='filename', stop_words='english', min_df=20)
>>> dtm = vectorizer.fit_transform(talks).toarray()
```

```python
>>> model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
>>> model.fit(X)
```
