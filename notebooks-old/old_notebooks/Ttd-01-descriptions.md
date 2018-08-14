
# Ted Talk Descriptions <a id='contents'></a>

This notebook has the following sections:

* [`BeautifulSoup`](#beautifulsoup) - code used to clean the downloaded files and save to `csv`. (The file is already saved, so this cell does not need to be run again.)
* [`pandas`](#pandas) - run this cell to load the csv file into three lists: titles, views, descriptions. 
* [Jaccard Matrices](#jaccard) - 
 - [John]() - 
 - [Katherine](#katherine) - this one needs to be run to load the matrix into memory.

## `BeautifulSoup`<a id='beautifulsoup'></a>


```python
import re
import csv
import os
from bs4 import BeautifulSoup

# title: <span class="player-hero__title__content">
# description: <p class="talk-description" lang="en">
# views: <span class="talk-sharing__value"> 96,013 </span>


def parse(soup):
    # both title and views are can be parsed in separate tags.
    title = soup.find('span', {'class' : "player-hero__title__content"}).text.strip('\n')
    views = soup.find('span', {'class' : "talk-sharing__value"}).text.strip('\n')
    descr = soup.find('p', {'class' : "talk-description"}).text.strip('\n')
    return title.strip(), views, descr

def to_csv(pth, out):
    # open file to write to.
    with open(out, "w") as out:
        # create csv.writer.
        wr = csv.writer(out)
        # write our headers.
        wr.writerow(["title", "views", "descr"])
        # get all our html files.
        for html in os.listdir(pth):
            with open(os.path.join(pth, html)) as f:
                print(html)
                # parse the file and write the data to a row.
                wr.writerow(parse(BeautifulSoup(f, "lxml")))

to_csv("./html_files/descriptions/",".</data/descriptions-2.csv") # This is to the test directory!
```

## `pandas`<a id='pandas'></a>


```python
import pandas
colnames = ['title', 'views' , 'descr']
data = pandas.read_csv('./data/descriptions-2.csv', names=colnames)
titles = data.title.tolist()
views = data.views.tolist()
descriptions = data.descr.tolist()
```


```python
len(titles)
```

## Jaccard Trials <a id='jaccard'></a>

### `sklearn`


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_similarity_score

tokenize = lambda doc: doc.lower().split()
 
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(descriptions)
#print(sklearn_representation[0])
```


```python
# To get a list of punctuation:

print(string.punctuation)
```


```python
# To find out the ordinal of a particular vexing character:

somewords = descriptions[2]
print(ord(somewords(16))
```

### John's Attempt at a Jaccard Matrix <a id='john'></a>


```python
import string
from stop_words import get_stop_words


def ourtokens(ourstring):
    
    stoplist = set(get_stop_words('en'))
    finalList = []
    
    wordList = ourstring.lower().split()
    for i in range(len(wordList)):
        #wordList[i] = re.sub('[^a-zA-Z\']', '', wordList[i]).strip(chr(8212)) 
        #NOTE: the above left spaces and added empty strings
        
        no_punc = wordList[i].strip(string.punctuation) #remove most punctuation
        no_emphwhatever = no_punc.strip(chr(8212)) # remove that weirdness
        no_num = no_emphwhatever.strip(string.digits) #remove numbers
        
        if (len(no_num) > 0) and (no_num not in stoplist): # Requires stop_words
            # First conditional stops empty strings from being added
            finalList.append(no_num)
            
    return finalList
```


```python
def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    #print(intersection)
    union = set(query).union(set(document))
    #print(union)
    return len(intersection)/len(union)

jaccard_similarity(ourtokens(descriptions[1]), ourtokens(descriptions[2]))
```




    0.021739130434782608



### Katherine's Kreation of a Jaccard Matrix <a id='katherine'></a> [[contents](#contents)]


```python
# Create des_word_lists 
des_word_lists = []
for i in range(len(descriptions)):
    # Create list of words for each description
    words = ourtokens(descriptions[i])
    des_word_lists.append({'descriptions': descriptions[i], 'words': words})
    
    # Tells you where you are in the rows
    if (i % 100) == 0:
        print(str(i) + " rows completed")
```

    0 rows completed
    100 rows completed
    200 rows completed
    300 rows completed
    400 rows completed
    500 rows completed
    600 rows completed
    700 rows completed
    800 rows completed
    900 rows completed
    1000 rows completed
    1100 rows completed
    1200 rows completed
    1300 rows completed
    1400 rows completed
    1500 rows completed
    1600 rows completed
    1700 rows completed
    1800 rows completed
    1900 rows completed
    2000 rows completed
    2100 rows completed
    2200 rows completed



```python
# Create the new CSV
import csv

with open('.</data/desPlusWords-2.csv', 'w') as desfile:
    fields = ['descriptions', 'words']
    writer = csv.DictWriter(desfile, fieldnames = fields)
    
    writer.writeheader()
    writer.writerows(des_word_lists)
```


```python
import numpy
import pandas

# Load deswordlists from CSV
colnames = ['descriptions', 'words']
data = pandas.read_csv('./data/desPlusWords-2.csv', names=colnames)
descriptions = data.descriptions.tolist()
words = data.words.tolist()
```


```python
Ndes = len(des_word_lists)

# From http://stackoverflow.com/questions/568962/how-do-i-create-an-empty-array-matrix-in-numpy
jac_mat = numpy.zeros(shape=(Ndes,Ndes))

for i in range(Ndes):
    if (i % 100) == 0:
        print(str(i) + " rows completed")  
    
    # Start the pairwise computations
    
    for j in range((i+1),Ndes):
        # Pull the ith and jth document
        doc_i = des_word_lists[i]['words']
        doc_j = des_word_lists[j]['words']
        
        # Get the Jaccard similarity
        jac_ij = jaccard_similarity(doc_i, doc_j)
        
        # Since the Jaccard will be the same between i and j as it will between
        # j and i, we set JAC_MAT[i,j] and JAC_MAT[j,i] to be the same value
        jac_mat[i,j] = jac_ij
        jac_mat[j,i] = jac_ij
```

    0 rows completed
    100 rows completed
    200 rows completed
    300 rows completed
    400 rows completed
    500 rows completed
    600 rows completed
    700 rows completed
    800 rows completed
    900 rows completed
    1000 rows completed
    1100 rows completed
    1200 rows completed
    1300 rows completed
    1400 rows completed
    1500 rows completed
    1600 rows completed
    1700 rows completed
    1800 rows completed
    1900 rows completed
    2000 rows completed
    2100 rows completed
    2200 rows completed



```python
# This block find the maximum for the matrix

# Initialize the max to be zero. 
mat_max = 0

# Loop over all the rows
for i in range(Ndes):
    # Find the maximum for each row
    row_max = max(jac_mat[i])
    
    # Check if the current row's maximum is higher than the current MAT_MAX.
    # If the row maximum is bigger, then set MAT_MAX to the row maximum.
    if row_max > mat_max:
        mat_max = row_max

print(mat_max)
```


```python
jac_mean = jac_mat.mean()
print(jac_mean, mat_max)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-1-4b4cb161c9a0> in <module>()
    ----> 1 jac_mean = jac_mat.mean()
          2 print(jac_mean, mat_max)


    NameError: name 'jac_mat' is not defined



```python
print("The mean is {}, and the max is {}.".format(jac_mean, mat_max))
```

### A Little Exploration


```python
micro_jac = jac_mat[0:10,0:10]
```


```python
len(micro_jac)
```




    10



## The Network Woods


```python
import networkx as nx

micro = nx.from_numpy_matrix(micro_jac)
```


```python
print(len(G.nodes()), len(G.edges()))
```

    10 10



```python
import matplotlib.pyplot as plt

pos = nx.spring_layout(micro)
nx.draw(micro, pos)
plt.savefig('./outputs/micro_jac.png')
```


```python
# convert array to DF
# add node labels to the DF
# save DF to CSV
# nx.write_gexf(G,"descr-jac.gexf")
nx.write_weighted_edgelist(G, "./outputs/jacmat.edges", delimiter=',', encoding='utf-8')
```


```python

```

#### Trimming a Graph


```python
remove = [node for node, degree in G.degree().items() if degree <= 2]
gmt2 = G.remove_nodes_from(remove)
print(len(gmt2.nodes()), len(gmt2.edges()))
```


```python
remove = [node for node, degree in G.degree().items() if degree <= 1]
gmt1 = G.remove_nodes_from(remove)
#print(len(gmt1.nodes()), len(gmt1.edges()))
gmt1.nodes()
```


```python
len(G.nodes())
```




    2210




```python
# print([node for node, degree in G.degree().items() if degree > 1])
[node for node, degree in G.degree().items() if degree > 2]
```

## Heat Map


```python
# SEE: http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor

import matplotlib.pyplot as plt
import numpy as np
column_labels = titles
row_labels = titles
fig, ax = plt.subplots()
heatmap = ax.pcolor(jac_mat, cmap=plt.cm.jet)
colorbar()
# OPTIONS to consider later
# put the major ticks at the middle of each cell
#ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
#ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

# want a more natural, table-like display
#ax.invert_yaxis()
#ax.xaxis.tick_top()

#ax.set_xticklabels(row_labels, minor=False)
#ax.set_yticklabels(column_labels, minor=False)
plt.show()
```


```python

```

## `networkx` sandbox


```python
# pg = play graph
pg = nx.Graph()
pg.add_edge(1,2)
pg.add_edge(1,3)
pg.add_edge(1,4)
pg.add_edge(2,3)
pg.add_edge(2,4)
```


```python
pg.degree()
```


```python
nx.draw(pg)
```


```python
remove = [node for node, degree in pg.degree().items() if degree <= 2]
print(remove)
```


```python
pg.nodes()
```


```python
pg.edges()
```


```python
pg.remove_nodes_from(remove)
print(pg.nodes(), pg.edges(), pg.degree())
```

### Graph Drawing Functionality


```python
# From: https://www.udacity.com/wiki/creating-network-graphs-with-python

def draw_graph(graph, labels=None, graph_layout='shell',
               node_size=1600, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    # create networkx graph
    G=nx.Graph()

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(G)
    else:
        graph_pos=nx.shell_layout(G)

    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size, 
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G,graph_pos,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
                            font_family=text_font)

    if labels is None:
        labels = range(len(graph))

    edge_labels = dict(zip(graph, labels))
    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels, 
                                 label_pos=edge_text_pos)

    # show graph
    plt.show()

graph = [(0, 1), (1, 5), (1, 7), (4, 5), (4, 8), (1, 6), (3, 7), (5, 9),
         (2, 4), (0, 4), (2, 5), (3, 6), (8, 9)]

# you may name your edge labels
labels = map(chr, range(65, 65+len(graph)))
#draw_graph(graph, labels)

# if edge labels is not specified, numeric labels (0, 1, 2...) will be used
draw_graph(graph)
```


```python
import numpy
import pandas
import string
from stop_words import get_stop_words


# Load descriptions and titles
colnames = ['title', 'views' , 'descr']
data = pandas.read_csv('./data/descriptions-2.csv', names=colnames)
titles = data.title.tolist()
views = data.views.tolist()
descriptions = data.descr.tolist()

# Load deswordlists from CSV
colnames = ['descriptions', 'words']
data = pandas.read_csv('./data/desPlusWords-2.csv', names=colnames)
descriptions = data.descriptions.tolist()
words = data.words.tolist()


# A couple of functions

def ourtokens(ourstring):
    
    stoplist = set(get_stop_words('en'))
    finalList = []
    
    wordList = ourstring.lower().split()
    for i in range(len(wordList)):
        #wordList[i] = re.sub('[^a-zA-Z\']', '', wordList[i]).strip(chr(8212)) 
        #NOTE: the above left spaces and added empty strings
        no_punc = wordList[i].strip(string.punctuation) #remove most punctuation
        no_emphwhatever = no_punc.strip(chr(8212)) # remove that weirdness
        no_num = no_emphwhatever.strip(string.digits) #remove numbers
        if (len(no_num) > 0) and (no_num not in stoplist): # Requires stop_words
            # First conditional stops empty strings from being added
            finalList.append(no_num)            
    return finalList

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    #print(intersection)
    union = set(query).union(set(document))
    #print(union)
    return len(intersection)/len(union)

# Create JACCARD MATRIX
# From http://stackoverflow.com/questions/568962/how-do-i-create-an-empty-array-matrix-in-numpy
Ndes = len(des_word_lists)
jac_mat = numpy.zeros(shape=(Ndes,Ndes))
for i in range(Ndes):
    if (i % 100) == 0:
        print(str(i) + " rows completed")    
    # Start the pairwise computations
    for j in range((i+1),Ndes):
        # Pull the ith and jth document
        doc_i = des_word_lists[i]['words']
        doc_j = des_word_lists[j]['words']
        # Get the Jaccard similarity
        jac_ij = jaccard_similarity(doc_i, doc_j)
        # Since the Jaccard will be the same between i and j as it will between
        # j and i, we set JAC_MAT[i,j] and JAC_MAT[j,i] to be the same value
        jac_mat[i,j] = jac_ij
        jac_mat[j,i] = jac_ij
```
