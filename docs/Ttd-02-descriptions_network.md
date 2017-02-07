
## The Matrix


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

# Create des_word_lists 
des_word_lists = []
for i in range(len(descriptions)):
    # Create list of words for each description
    words = ourtokens(descriptions[i])
    des_word_lists.append({'descriptions': descriptions[i], 'words': words})
    
    # Tells you where you are in the rows
    if (i % 100) == 0:
        print(str(i) + " rows completed")
        
# Create JACCARD MATRIX
# From http://stackoverflow.com/questions/568962/how-do-i-create-an-empty-array-matrix-in-numpy
Ndes = len(des_word_lists)
jac_mat = numpy.zeros(shape=(Ndes,Ndes))
thresh_JM = numpy.zeros(shape=(Ndes,Ndes))
jac_lst = []
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
        
        if jac_ij > 0.06:
            thresh_JM[i,j] = jac_ij
            thresh_JM[j,i] = jac_ij
        
        # Get all the non-zero Jaccard values
        #if jac_ij != 0:
            #jac_lst.append(jac_ij)
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
>>> import numpy as np
...
>>> sort_jac_lst = sorted(jac_lst)
>>> np.histogram(jac_lst, bins = 5)
import numpy as np

sort_jac_lst = sorted(jac_lst)
np.histogram(jac_lst, bins = 5)
```

    (array([0, 0, 0, 0, 0]), array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ]))

```python


#jac_mat[0:8, 0:10]
thresh_JM[0:8, 0:10]
numpy.savetxt("./Thresh05.csv",thresh_JM, fmt = '%1.5f', delimiter=",")
```

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

    0.304347826087

## Network


```python
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# Create a network from the matrix
G = nx.from_numpy_matrix(thresh_JM)
```


```python
Gdegree = nx.average_degree_connectivity(G)
print(Gdegree)
```

    {0: 0.0, 1: 21.370786516853933, 2: 18.387096774193548, 3: 25.291666666666668, 4: 27.982954545454547, 5: 25.93846153846154, 6: 25.373333333333335, 7: 27.2404181184669, 8: 28.44396551724138, 9: 27.962962962962962, 10: 28.89814814814815, 11: 30.183441558441558, 12: 31.730654761904763, 13: 29.952662721893493, 14: 31.207792207792206, 15: 31.830065359477125, 16: 33.125, 17: 30.56644880174292, 18: 33.583333333333336, 19: 31.810526315789474, 20: 29.520588235294117, 21: 33.22278911564626, 22: 35.35016835016835, 23: 31.944664031620555, 24: 32.82692307692308, 25: 36.4375, 26: 32.24038461538461, 27: 33.64522417153996, 28: 37.42307692307692, 29: 32.3132183908046, 30: 34.1875, 31: 34.62442396313364, 32: 34.08894230769231, 33: 33.25252525252525, 34: 31.91764705882353, 35: 33.41428571428571, 36: 35.295138888888886, 37: 33.59121621621622, 38: 33.9671052631579, 39: 32.04761904761905, 40: 34.80833333333333, 41: 37.423780487804876, 42: 33.96666666666667, 43: 35.41860465116279, 44: 33.63636363636363, 45: 36.6395061728395, 46: 34.0, 47: 35.43262411347518, 48: 38.954166666666666, 49: 36.425170068027214, 50: 35.296, 51: 37.21078431372549, 52: 33.40384615384615, 53: 34.503144654088054, 54: 37.574074074074076, 55: 32.14545454545455, 56: 37.42261904761905, 57: 33.45614035087719, 58: 33.827586206896555, 59: 37.07627118644068, 60: 32.55, 61: 41.59016393442623, 64: 33.375, 65: 36.12307692307692, 66: 36.28787878787879, 68: 35.01470588235294, 71: 35.15492957746479, 140: 32.607142857142854, 74: 34.2972972972973, 75: 41.13333333333333, 77: 33.675324675324674, 80: 37.85, 83: 34.46987951807229, 87: 34.758620689655174, 91: 32.43956043956044, 93: 33.166666666666664, 94: 33.91489361702128, 96: 38.4375, 99: 34.5959595959596, 100: 36.39, 103: 32.25242718446602, 106: 35.39622641509434, 107: 38.177570093457945}

```python


remove = [node for node, degree in G.degree().items() if degree == 40]
G.remove_nodes_from(remove)
#print(remove)
print(len(G.nodes()), len(G.edges()))
```

    1396 11132

```python


# Draw the graph

pos = nx.spring_layout(G, k=0.20, iterations=10)
fig = plt.figure(figsize=(15,15))
nx.draw(G, pos)
plt.show()
```

![png](TT_descriptions_network_files/TT_descriptions_network_10_0.png)



```python
# Save edge list
# write_edgelist(G, path, delimiter=',', data=True, encoding='utf-8')
nx.write_weighted_edgelist(G, './outputs/thresh05.csv', delimiter=',', encoding='utf-8')
```

### Prune the Network


```python
# SCANNING
# Gdegree = nx.average_degree_connectivity(G)
print(Gdegree)

# PRUNING                            
# remove = [node for node, degree in G.degree().items() if degree <= 4]
print(len(remove))
# gmt2 = G.remove_nodes_from(remove)
# print(len(gmt2.nodes()), len(gmt2.edges()))
```


```python
import csv

with open('./outputs/desc-labels.csv', 'w', newline='\n') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(titles)
```


```python
type(titles)
```

    list

```python


# Example of writing JSON format graph data and using the D3 Javascript library
# to produce an HTML/Javascript drawing.
#    Copyright (C) 2011-2012 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.

__author__ = """Aric Hagberg <aric.hagberg@gmail.com>"""
import json
import networkx as nx
from networkx.readwrite import json_graph
import http_server

G = nx.barbell_graph(6,3)
# this d3 example uses the name attribute for the mouse-hover value,
# so add a name to each node
for n in G:
    G.node[n]['name'] = n
# write json formatted data
d = json_graph.node_link_data(G) # node-link format to serialize
# write json
json.dump(d, open('force/force.json','w'))
print('Wrote node-link JSON data to force/force.json')
# open URL in running web browser
http_server.load_url('force/force.html')
print('Or copy all files in force/ to webserver and load force/force.html')
```
