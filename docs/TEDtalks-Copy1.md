
# TED Talk Transcripts

## First Draft: Working with BeautifulSoup

A bit of experimentation led to the development of this first script for grabbing a file and getting just the AUTHOR, TITLE, DATE, LENGTH, and TEXT.

```python
import glob, re, csv
from bs4 import BeautifulSoup as soup                                                     

the_file = "/Users/john/Code/tedtalks/test/transcript?language=en.0"
holding = soup(open(the_file).read(), "lxml")
at = holding.find("title").text
author = at[0:at.find(':')]
title  = at[at.find(":")+1 : at.find("|") ]
date = re.sub('[^a-zA-Z0-9]',' ', holding.select_one("span.meta__val").text)
length_data = holding.find_all('data', {'class' : 'talk-transcript__para__time'})
(m, s) = ([x.get_text().strip("\n\r") 
      for x in length_data if re.search(r"(?s)\d{2}:\d{2}", 
                                        x.get_text().strip("\n\r"))][-1]).split(':')
length = int(m) * 60 + int(s)
firstpass = re.sub(r'\([^)]*\)', '', holding.find('div', class_ = 'talk-transcript__body').text)
text = re.sub('[^a-zA-Z\.\']',' ', firstpass)
data = [str(author), str(title)]
# print(data)
with open("./output.csv", "w", newline = "") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for item in data:
            writer.writerow(item)
```

After getting that to work, I imported some boilerplate that has worked for me in the past:

```python
file_list = glob.glob('/Users/john/Code/tedtalks/test/*') # produces list
print(file_list)
```

Which produces a list of files just fine:

```python
['/Users/john/Code/tedtalks/test/transcript?language=en.0', '/Users/john/Code/tedtalks/test/transcript?language=en.1', '/Users/john/Code/tedtalks/test/transcript?language=en.2']
```

But handing that off to the initial script proved very tricky. It was clearly time to learn how to `define` functions. With more gratitude than I can express, [Padraic Cunningham][] not only developed the script below, but was also very patient in diagnosing a particular problem I encountered.

The script below is available in the repo as `talks_to_csv.py`.

[Padraic Cunningham]: http://chat.stackoverflow.com/users/2141635


```python
import re
import csv
import os
from bs4 import BeautifulSoup

def parse(the_soup):
    # both title and author are can be parsed in separate tags.
    author = the_soup.select_one("h4.h12.talk-link__speaker").text.encode("utf-8")
    title = the_soup.select_one("h4.h9.m5").text
    # just need to strip the text from the date string, no regex needed.
    date = the_soup.select_one("span.meta__val").text.strip()      
    # we want the last time which is the talk-transcript__para__time previous to the footer.
    mn, sec = map(int, the_soup.select_one("footer.footer").find_previous("data", {
    "class": "talk-transcript__para__time"}).text.split(":"))
    length = (mn * 60 + sec)        
    # to ignore (Applause) etc.. we can just pull from the actual text fragment checking for (
    text = " ".join(d.text for d in the_soup.select("span.talk-transcript__fragment") if not d.text.startswith("("))        
    # clean the text
    text = re.sub('[^a-zA-Z\.\']', ' ', text)
    return  author.strip(), title.strip(), date, length, text


def to_csv(pth, out):
    # open file to write to.
    with open(out, "w") as out:
        # create csv.writer. 
        wr = csv.writer(out)
        # write our headers.
        wr.writerow(["author", "title", "date", "length", "text"])
        # get all our html files.
        for html in os.listdir(pth):
            with open(os.path.join(pth, html)) as f:
                # parse the file are write the data to a row.
                wr.writerow(parse(BeautifulSoup(f, "lxml")))
                
to_csv("./test","test.csv")
```

Fix below is to remove parentheses and numbers.


```python
import re
import csv
import os
from bs4 import BeautifulSoup


def parse(soup):
    # both title and author are can be parsed in separate tags.
    author = soup.select_one("h4.h12.talk-link__speaker").text
    title = soup.select_one("h4.h9.m5").text
    # just need to strip the text from the date string, no regex needed.
    date = soup.select_one("span.meta__val").text.strip()
    # we want the last time which is the talk-transcript__para__time previous to the footer.
    mn, sec = map(int, soup.select_one("footer.footer").find_previous("data", {
        "class": "talk-transcript__para__time"}).text.split(":"))
    length = (mn * 60 + sec)
    # to ignore time etc.. we can just pull from the actual text fragment and remove noise i.e (Applause).
    text = re.sub(r'\([^)]*\)',"", " ".join(d.text for d in soup.select("span.talk-transcript__fragment")))
    return author.strip(), title.strip(), date, length, re.sub('[^a-zA-Z\.\']', ' ', text)

def to_csv(pth, out):
    # open file to write to.
    with open(out, "w") as out:
        # create csv.writer.
        wr = csv.writer(out)
        # write our headers.
        wr.writerow(["author", "title", "date", "length", "text"])
        # get all our html files.
        for html in os.listdir(pth):
            with open(os.path.join(pth, html)) as f:
                print(html)
                # parse the file are write the data to a row.
                wr.writerow(parse(BeautifulSoup(f, "lxml")))

to_csv("./talks","talks.csv") # This is to the test directory!
```

Next task is to read all the texts *qua* texts and to be able to do basic things like word frequency and topic modeling...


```python
import pandas
colnames = ['author', 'title', 'date' , 'length', 'text']
data = pandas.read_csv('./talks-v1b.csv', names=colnames)
talks = data.text.tolist()
# importing all the talks here. If we want to test, we should import 
# talks from 2006 - 2015 and then train and test on 2016
```


```python
len(talks)
```




    2113




```python
len(talks[1])
```




    17830




```python
i = 0
no_good = []
for talk in talks: 
    A = type(talk)
    B = type('string or something')
    if A != B:
        no_good.append(i)
    i = i + 1
```


```python
no_good
```




    [185, 398, 513, 877, 1015, 1100, 2011]




```python
for index in sorted(no_good, reverse=True):
    del talks[index]
```


```python
len(talks)
```




    2106




```python
# Re-checking for anything NOT a string
i = 0
still_no_good = []
for talk in talks: 
    A = type(talk)
    B = type('string or something')
    if A != B:
        still_no_good.append(i)
    i = i + 1
```


```python
len(talks)
```




    2106




```python
from stop_words import get_stop_words
from gensim import corpora, models, similarities

# remove common words and tokenize
stoplist = set(get_stop_words('en'))
texts = [[word for word in talk.lower().split() if word not in stoplist]
         for talk in talks]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
         for text in texts]
```


```python
len(texts[1])
```




    1584




```python
dictionary = corpora.Dictionary(texts)
dictionary.save('./talks.dict')
```


```python
print(dictionary)
# To see the assignments for the tokens:
# print(dictionary.token2id)
```

    Dictionary(47144 unique tokens: ['abolitionists', 'pizarro', 'downstairs', 'graded', 'thrones']...)



```python
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('./talks.mm', corpus) # Save corpus in Market Matrix format
# To load this corpus: corpus = corpora.MmCorpus('./talks.mm')
```


```python
corpora.BleiCorpus.serialize('./talks.lda-c', corpus) # To save in LDA-C format
```


```python
tfidf = models.TfidfModel(corpus) # to train a portion of the corpus
```


```python
corpus_tfidf = tfidf[corpus] # to transform the entire corpus
```


```python

```
