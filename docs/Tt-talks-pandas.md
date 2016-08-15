

```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib



```python
import pandas
colnames = ['author', 'title', 'date' , 'length', 'text']
data = pandas.read_csv('./data/talks-v1b.csv', names=colnames)
talks = data.text.tolist()
# importing all the talks here. If we want to test, we should import 
# talks from 2006 - 2015 and then train and test on 2016
```

A `len(talks)` shows that we've got a list of 2113 strings, and typing in a random number, `talk[#]` gives me the contents of a talk. It also revealed that some talks, for whatever reason, have no content. A quick script from Katherine gave us a way to find those strings. (The content of those strings is `'n'`, e.g. `talk[185]`.


```python
i = 0
no_good = []
for talk in talks: 
    A = type(talk)
    B = type('string or something')
    if A != B:
        no_good.append(i)
    i = i + 1
    
print(no_good)
```

    [185, 398, 513, 877, 1015, 1100, 2011]


The next bit of scripting, first, removes the `empty` talks but does so in reverse order in order to avoid disturbing the index order -- n.b., if we re-associate labels from elsewhere in the CSV, we will need to remove the same rows from that list as well.

After purging, we check to see that we are indeed "seven down" and then we re-check for empty strings. The result is an empty list. Good.


```python
for index in sorted(no_good, reverse=True):
    del talks[index]
    
print(len(talks))

# Re-checking for anything NOT a string
i = 0
still_no_good = []
for talk in talks: 
    A = type(talk)
    B = type('string or something')
    if A != B:
        still_no_good.append(i)
    i = i + 1

print(still_no_good)
```

    2106
    []



```python
data[:5]
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
data.ix['Averting the climate crisis'] # ERROR: "KeyError"
```


```python
data.columns
```




    Index(['author', 'title', 'date', 'length', 'text'], dtype='object')




```python
data.author.tail(8)
```




    2105           Richard Seymour
    2106               Ian Ritchie
    2107              Pamela Meyer
    2108              Jae Rhim Lee
    2109                Bunker Roy
    2110       Justin Hall-Tipping
    2111    Guy-Philippe Goldstein
    2112               Todd Kuiken
    Name: author, dtype: object




```python
data.length.std()
```




    352.79834554674466




```python
data.length.mean()
```




    785.5281590156176




```python
data.length.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10de16fd0>




![png](Tt-talks-pandas_files/Tt-talks-pandas_12_1.png)

