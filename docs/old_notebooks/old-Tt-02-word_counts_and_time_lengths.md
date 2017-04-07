

```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib



```python
import re
import pandas
import matplotlib.pyplot as plt
import numpy as np
import ggplot as gg

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# global MPL settings
# (http://bikulov.org/blog/2013/10/03/creation-of-paper-ready-plots-with-matlotlib/)

def init_plotting():
    plt.rcParams['figure.figsize'] = (15, 9)
    plt.rcParams['font.size'] = 12
#    plt.rcParams['font.family'] = 'Avenir Next'
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['axes.linewidth'] = 1

    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

# init_plotting()

# Limiting chart to where the data is
plt.ylim(0, 3000)  
plt.xlim(0, 2200)  
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

colnames = ['author', 'title', 'date' , 'length', 'text']
data = pandas.read_csv('../data/talks_2.csv', names=colnames)
texts = data.text.tolist()

# It's entirely possible to convert a column into a list:
# lengths = data.length.tolist()
# but I am experimenting with leaving things in the pandas dataframe.
# The results of most operations, e.g. len(lengths) and len(data.length)
# are the same (2113).
```


![png](output_1_0.png)



```python
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>author</th>
      <th>title</th>
      <th>date</th>
      <th>length</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Al Gore</td>
      <td>Averting the climate crisis</td>
      <td>Jun 2006</td>
      <td>957</td>
      <td>Thank you so much  Chris. And it's truly a gre...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>David Pogue</td>
      <td>Simplicity sells</td>
      <td>Jun 2006</td>
      <td>1271</td>
      <td>Hello voice mail  my old friend. I've called f...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cameron Sinclair</td>
      <td>My wish: A call for open-source architecture</td>
      <td>Jul 2006</td>
      <td>1398</td>
      <td>I'm going to take you on a journey very quickl...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sergey Brin + Larry Page</td>
      <td>The genesis of Google</td>
      <td>May 2007</td>
      <td>1205</td>
      <td>Sergey Brin  I want to discuss a question I kn...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nathalie Miebach</td>
      <td>Art made of storms</td>
      <td>Oct 2011</td>
      <td>247</td>
      <td>What you just heard are the interactions of ba...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Simplest solution (but I can't figure out how to sort -- see graph below)
data.length.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1038d5b70>




![png](output_3_1.png)



```python
# Sorted graph using MPL
length_sorted = sorted(data.length)
plt.plot(length_sorted)

plt.xlabel('TEDtalks')
plt.ylabel('Length in Seconds')
plt.title('How Long is a TEDtalk?')
plt.grid(True)
plt.show()
```


![png](output_4_0.png)



```python
# The graph reveals we have a number of items that are, in fact, zero that
# will affect other mathematical operations: mean, median, etc. How many are
# in that flat head and how many in the spiked tail? 
# (See ggplot histogram below.)

print(length_sorted[0:100])
print(length_sorted[-20:])
```

    [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 16, 25, 119, 123, 126, 137, 142, 143, 153, 154, 155, 156, 158, 163, 164, 165, 166, 167, 172, 173, 175, 177, 177, 178, 179, 179, 182, 182, 183, 184, 188, 189, 191, 191, 193, 193, 193, 193, 195, 196, 196, 197, 198, 199, 200, 201, 201, 203, 204, 204, 205, 211, 212, 213]
    [1652, 1654, 1664, 1664, 1681, 1692, 1739, 1747, 1752, 1760, 1849, 1885, 1910, 1967, 2000, 2001, 2061, 2089, 2201, 2663]



```python
len(re.findall("[a-zA-Z_]+", data.text[0]))
```




    2176




```python
# First attempt at speed using built-in pandas plotting:
import re
len(re.findall("[a-zA-Z_]+", data.text))/(data.length).plot()
```


```python
# Problem: some items in list of texts are not strings. 
# We need to get rid of them or replace them:

def xstr(s):
    if s is None:
        return ''
    else:
        return str(s)

# With any luck, this list comprehension will work:
only_texts = [ xstr(text) for text in texts ]
```


```python
len(only_texts)
```




    2092




```python
# Now will this work?
counts = [len(re.findall("[a-zA-Z_]+", text)) for text in only_texts]
```


```python
counts[0:5]
```




    [2176, 3417, 3700, 3671, 659]




```python
# Success with the apparent clean-up: 
# less clear is how to make sure the word count and length of a talk MATCH
# I'm not sure if that's happening here: this should be a unidimensional graph.

wpm = []
for count in counts:
    wpm_ = count / (data.length/60)
    wpm.append(wpm_)
```


```python
wpm[0][0:5] # reveals that we have 2000 data points for each item in the list. 
```




    0    136.426332
    1    102.722266
    2     93.390558
    3    108.348548
    4    528.582996
    Name: length, dtype: float64




```python
lengths = data.length / 60
```


```python
lengths.head()
```




    0    15.950000
    1    21.183333
    2    23.300000
    3    20.083333
    4     4.116667
    Name: length, dtype: float64




```python
len(lengths)
```




    2092




```python
init_plotting()

plt.plot(counts, lengths, 'ro')
plt.xlabel('Length of Talk in Words')
plt.ylabel('Length of Talk in Minutes')
plt.title('How Fast is a TEDtalk?: Words per Minute')
plt.grid(True)
plt.show()
```


![png](output_17_0.png)



```python
# Third attempt: add a column with word count into pandas dataframe

# data['word_count'] = data['text'].map(len(re.findall("[a-zA-Z_]+", data.text)))
# data['word_count'] =  data['text'].apply(lambda x: len(re.findall("[a-zA-Z_]+", x)))

word_counts = []
for item in data.text: # data.text = string not list
    wordcount = len(re.findall("[a-zA-Z_]+", item)
#    word_counts.append(wordcount)
    
```


```python
word_counts = []
for text in texts:
    count = len(re.findall(r"[a-zA-Z_]+", text))
    word_counts.append(count)
```


```python
texts[0].split()[:10]
```


```python
type(data.text[0])
```


```python
for text in texts[185]:
    print(text.split()[:5])
```


```python
def count_words2(cell):
    try:
        len(cell.split())
    except AttributeError:
        return cell

data['word_count'] = data['text'].apply(count_words2)
```


```python
data.head()
```


```python
# Make a list, turn it into a giant string
talks = df.text.tolist()
alltalks = " ".join(str(item) for item in talks) # Solves pbm of floats in talks
all_words = re.sub(r"[^\w\d'\s]+",'',alltalks).lower() # Remove all punctuation save apostrophes

# Tokenize on whitespace with NLTK
from nltk.tokenize import WhitespaceTokenizer
tt_tokens = WhitespaceTokenizer().tokenize(all_words)

# Build a dictionary of words and their frequency in the corpus
tt_freq = {}
for word in tt_tokens:
    try:
        tt_freq[word] += 1
    except: 
        tt_freq[word] = 1

# Convert dictionary into a list of tuples
tt_freq_list = [(val, key) for key, val in tt_freq.items()]

# Sort with most frequent words at top
tt_freq_list.sort(reverse=True)

# Write to file
with open('../outputs/tt_freq.csv', 'w') as f:
        wtr = csv.writer(f)
        wtr.writerows(sorted(tt_freq_list, reverse=True))
        f.close()
```

## Experiments with ggplot


```python
gg.ggplot(data, gg.aes('length')) + gg.geom_density()

# The resulting graph does seem to represent the frequency of certain lengths
# within the corpus. However, I cannot make sense of the numbers on the Y axis.
# E.g., "0.0000 - 0.0014": possibly percentages? (Check ggplot docs!)
```


```python
gg.ggplot(data, gg.aes('length')) + gg.geom_histogram()
```


```python
gg.ggplot(data, gg.aes('length')) + gg.geom_point()
```


```python
# An early attempt to convert talk lengths from seconds to minutes and 
# seconds -- I think I prefer just dividing by sixty and rounding to a tenth.

def minsecs (seconds):
    m, s = divmod(seconds, 60)
    return m, s

print("{}{}".format(m, s = minsecs(data.length.mean())))

#print("The average talk is {} minutes and {} seconds.".format(minsecs(data.length.mean())))
#print("The median TEDtalk length is {} seconds".format(data.length.median()))
```


```python
np.average,
np.median,
np.mode,
np.arange
# average(lengths)
# print("The average talk is {} minutes long.".format(average/60))
mini(lengths)
maxi(lengths)
```


```python
data.length.max()
```


```python
from statistics import mode

mode(data.length)
```


```python
from collections import Counter
counted = Counter(lengths)
counted.most_common()   # Returns all unique items and their counts
counted.most_common(1)  # Returns the highest occurring item
```


```python
print(lengths.sort())
```


```python
talks = data.text.tolist()
authors = data.author.tolist()
dates = data.date.tolist()

# Combining year with presenter for citation
authordate = [author+" "+year for author, year in zip(authors, years)]
```


```python
import ggplot
```


```python
# Type-to-Token Ratio (TTR) = crude measure of diversity (stupid at this scope)
lex_diversity = round(len(tt_freq_list) / len(all_words), 4)
print("Lexical diversity: {}".format(lex_diversity))
```
