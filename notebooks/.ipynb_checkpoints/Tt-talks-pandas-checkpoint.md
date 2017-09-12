```python
>>> %pylab inline
Populating the interactive namespace from numpy and matplotlib
```

```python
>>> import pandas
>>> colnames = ['author', 'title', 'date' , 'length', 'text']
>>> data = pandas.read_csv('../data/talks-v1b.csv', names=colnames)
>>> talks = data.text.tolist()
>>> # importing all the talks here. If we want to test, we should import
... # talks from 2006 - 2015 and then train and test on 2016
```

```python
>>> # If we want to delete the 7 talks with no text:
... i = 0
>>> no_good = []
>>> for talk in talks:
...     A = type(talk)
...     B = type('string or something')
...     if A != B:
...         no_good.append(i)
...     i = i + 1
...
>>> print(no_good)
[185, 398, 513, 877, 1015, 1100, 2011]
```

```python
>>> for index in sorted(no_good, reverse=True):
...     del talks[index]
```

```python
>>> data[0:5]
                     author                                         title  \
0                   Al Gore                   Averting the climate crisis   
1               David Pogue                              Simplicity sells   
2          Cameron Sinclair  My wish: A call for open-source architecture   
3  Sergey Brin + Larry Page                         The genesis of Google   
4          Nathalie Miebach                            Art made of storms   

       date  length                                               text  
0  Jun 2006     957  Thank you so much  Chris. And it's truly a gre...  
1  Jun 2006    1271  Hello voice mail  my old friend. I've called f...  
2  Jul 2006    1398  I'm going to take you on a journey very quickl...  
3  May 2007    1205  Sergey Brin  I want to discuss a question I kn...  
4  Oct 2011     247  What you just heard are the interactions of ba...
```

```python
>>> data.columns
Index(['author', 'title', 'date', 'length', 'text'], dtype='object')
```

```python
>>> data.author.tail(5)
2108              Jae Rhim Lee
2109                Bunker Roy
2110       Justin Hall-Tipping
2111    Guy-Philippe Goldstein
2112               Todd Kuiken
Name: author, dtype: object
```

```python
>>> data.length.std()
352.79834554674454
```

```python
>>> data.length.mean()
785.52815901561758
```

```python
>>> data.length.plot()
```

```python
>>> data.length.mode()
0    11
dtype: int64
```

```python
>>> type(data.length)
pandas.core.series.Series
```

```python
>>> lengths = data.length.tolist()
```

```python
>>> from statistics import *
...
>>> print("The mean is {}.".format(mean(lengths)))
>>> print("The median is {}.".format(median(lengths)))
>>> print("The 50th percentile is {}.".format(median_grouped(lengths, interval=10)))
>>> print("The mode is {}.".format(mode(sorted(lengths))))
The mean is 785.
The median is 838.
The 50th percentile is 835.5.
The mode is 11.
```

```python
>>> for talk in talks:
...     # turn text into list
...     # count items in list
...     # write count to a new list WORDS
...
>>> for item in words:
...     # divide item by length
...     # return quotient as wpm
```
