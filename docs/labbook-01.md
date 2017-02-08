# Lab Notes

Okay, we need an infrastructure both for drafting as well as for communicating with each other outside of emails, code, or video. It's one way to capture not only for you but also for me some of the things I encounter without having a lot of clutter in the Jupyter notebooks, if we pursue those as a collaborative option.

## TO DO: 

- [ ] Compare the most frequent words here against established stopword lists to see what a TED talk stoplist would look like.
- [ ] Create lexical diversity measure for all texts. Since lengths are fairly comparable, TTR will be, I think, good enough.
- [x] Create a list of all the empty or "bad" texts.

- [ ] See what affect stemming has -- though this shouldn't be a high priority since stemming seems to be debatable. >>> I decided to skip this in favor of seeing how things turn out unstemmed first.

## 2017-01-29

What I've been working on for the past few days is in preparation for attempting a topic model using the more established LDA instead of the NMF to see how well they compare -- with the understanding that since there is rarely a one-to-one matchup within either method, that there will be no such match across them.

Because LDA does not filter out common words on its own, the way the NMF method does, you have to start with a stoplist. I know we can begin with Blei's and a few other established lists, but I would also like to be able to compare that against our own results. My first thought was to build a dictionary of words and their frequency within the corpus. For convenience sake, I am using the NLTK. 

Just as a record of what I've done, here's the usual code for loading the talks from the CSV with everything in it:

```python
import pandas
import re

# Get all talks in a list & then into one string
colnames = ['author', 'title', 'date' , 'length', 'text']
df = pandas.read_csv('../data/talks-v1b.csv', names=colnames)
talks = df.text.tolist()
alltalks = " ".join(str(item) for item in talks) # Solves pbm of floats in talks

# Clean out all punctuation except apostrophes
all_words = re.sub(r"[^\w\d'\s]+",'',alltalks).lower()
```


We still need to identify which talks have floats for values and determine what impact, if any, it has on the project.

```python
import nltk

tt_tokens = nltk.word_tokenize(all_words)

tt_freq = {}
for word in tt_tokens:
    try:
        tt_freq[word] += 1
    except: 
        tt_freq[word] = 1
```

Using this method, the dictionary has 63426 entries. Most of those are going to be single-entry items or named entities, but I do think it's worth looking at them, as well as the high-frequency words that may not be a part of established stopword lists: I think it will be important to note those words which are specifically common to TED Talks.

I converted the dictionary to a list of tuples in order to be able to sort -- I see that there is a way to sort a dictionary in Python, but this is a way I know. Looking at the most common words, I see NLTK didn't get rid of punctuation: I cleared this up by removing punctuation earlier in the process, keeping the contractions (words with apostrophes), which the NLTK does not respect. 

**N.B.** I tried doing this simply with a regex expression that split on white spaces, but I am still seeing contractions split into different words. 

```python
tt_freq_list.sort(reverse=True)
tt_freq_list[0:20]

[(210294, 'the'),
 (151163, 'and'),
 (126887, 'to'),
 (116155, 'of'),
 (106547, 'a'),
 (96375, 'that'),
 (83740, 'i'),
 (78986, 'in'),
 (75643, 'it'),
 (71766, 'you'),
 (68573, 'we'),
 (65295, 'is'),
 (56535, "'s"),
 (49889, 'this'),
 (37525, 'so'),
 (33424, 'they'),
 (32231, 'was'),
 (30067, 'for'),
 (28869, 'are'),
 (28245, 'have')]
```

Keeping the apostrophes proved to be harder than I thought -- and I tried going a "pure Python" route and splitting only on white spaces, trying both of the following: 

```python
word_list = re.split('\s+', all_words)
word_list = all_words.split()
```

I still got: ` (56535, "'s"),`. (The good news is that the counts match.)

Okay, good news. The NLTK white space tokenizer works:

```python
from nltk.tokenize import WhitespaceTokenizer
white_words = WhitespaceTokenizer().tokenize(all_words)
```

I tried using Sci-Kit Learn's `CountVectorizer` but it requires a list of strings, not one string, **and** it does not like that some of the texts are floats. So, we'll save dealing with that when it comes to looking at this corpus as a corpus and not as one giant collection of words.

```python
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
word_counts = count_vect.fit_transform(talks)

ValueError: np.nan is an invalid document, expected byte or unicode string.
```

The final, working, script of the day produces the output we want:

```python

# Tokenize on whitespace
from nltk.tokenize import WhitespaceTokenizer
tt_tokens = WhitespaceTokenizer().tokenize(all_words)

# Build a dictionary of words and their frequency in the corpus
tt_freq = {}
for word in tt_tokens:
    try:
        tt_freq[word] += 1
    except: 
        tt_freq[word] = 1

# Build a list of tuples, sort, and see some results 
tt_freq_list = [(val, key) for key, val in tt_freq.items()]
tt_freq_list.sort(reverse=True)
tt_freq_list[0:20]
```



## 2017-01-31

A quick check of the length of the vocabulary set (`len(tt_freq_list)`) for the TED Talks using the NLTK method reveals a total vocabulary of unique words of 56,714.[^1] The total word count for the corpus (`len(all_words)`) is 24,117,574. A raw lexical diversity figure (Type-to-Token Ratio, or TTR) would be: 

```python
lex_diversity = round(len(tt_freq_list) / len(all_words), 4)
print("Lexical diversity: {}".format(lex_diversity))

Lexical diversity: 0.0024
``` 
This is for the entire corpus: it would be interesting to see that on a per-text basis and to see which texts are most/least diverse and also to see if their is any tendencies by gender, discipline, etc. (Can we determine a list of those *et cetera*s?)



## 2017-02-01

Results of counting stopword lists:

```python
stop_words has 174 words 
the NLTK has 153 words 
the TedTalk list has 302 words 
the MALLET list has 525 words 
the Blei list has 297 words
```

Again, I don't have notes for the TEDtalk list, so I'm going to throw this list out for the time being. (I am appending "_old" to the file name.)

A bit more work comparing stopword lists, and I have a base list determined:

```python
import re

tt_stoplist = re.split('\s+', open('../data/tt_stop.txt', 'r').read().lower())
print("tt_stop has {} words".format(len(tt_stoplist)))

tt_stop has 351 words
```

## 2017-02-02: Compiling a List of the "Bad" Texts

I am beginning with the following code that KK wrote to find those talks that were not strings:

```python
# We establish which talks are empty
i = 0
no_good = []
for talk in talks:
    A = type(talk)
    B = type('string or something')
    if A != B:
        no_good.append(i)
    i = i + 1

print(no_good)

[185, 398, 513, 877, 1015, 1100, 2011]
```

In `pandas` in addition to being able to filter rows, you can also select by position. It looks a lot like slicing in lists. (You can do this two dimensionally as well.) **N.B.** Another way to do this would have been to filter by `NaN` but I already had the list above so I went with this method.

```python
df.iloc[no_good]
```

author |	title	| date	| length	| text
-------|------------|-------|-----------|---------
185	| Quixotic Fusion	| Dancing with light	| Jun 2012	| 718	| NaN
398	| Bruno Maisonnier	| Dance, tiny robots!	| Feb 2013	| 74	| NaN
513	| Kenichi Ebina	    | My magic moves	    | Oct 2007	| 204	| NaN
877	| Aakash Odedra	    |A dance in a hurricane of paper, wind and light	| Dec 2014	| 573	| NaN
1015 | Joey Alexander	| An 11-year-old prodigy performs old-school jazz	| Jun 2015	| 372	| NaN
1100 | Kaki King | A musical escape into a world of light and color	| Nov 2015	| 671	| NaN
2011 | Robert Gupta + Joshua Roman	| On violin and cello, "Passacaglia"	| May 2011	| 526	| NaN

A visual inspection of the CSV file confirmed the lack of text in each of these rows. A quick check of the TED website found the following:

* 185: Yup, no words.
* 398: No transcript on website. 3:00 long.
* 513: No transcript. 3:28.
* 877: No transcript. 9:50.

I performed a similar operation to determine texts that might be too short to contribute to a topic model and other measures of texts. I simply ball-parked 1000 characters -- because in a previous moment of graphing I had seen numbers in the 200s -- and 21 texts turned up:

```python
j = 0
too_short = []
for talk in talks: 
    if len(str(talk)) < 1000:
        too_short.append(j)
    j = j + 1

print(len(too_short), too_short)

21 [115, 185, 331, 398, 513, 877, 982, 1015, 1100, 1299, 1342, 1427, 1641, 1846, 1852, 1937, 1947, 2011, 2028, 2080, 2102]
```

I inspected the CSV itself, `../data/talks-v1b.csv`, again, and these results are confirmed.

So, now, with a stopword list and the empty talks or the too short talks all identified, we are ready to proceed with a topic model...

***Before*** moving on, I decided to clean up the `docs/` directory. It's a mess of my own making. There's a fair amount of duplication in there, and, to be honest, the markdown version of the notebooks aren't working in terms of keeping the code and the text explanations clearly separated. (Plus, GitHub now supports the display of Jupyter Notebook pages.)

## 2017-02-03 - LDA Topic Model

The topic model work begins with our standard loading of a `pandas` dataframe from the CSV:

```python
import pandas
import re

# Create pandas dataframe
colnames = ['author', 'title', 'date' , 'length', 'text']
df = pandas.read_csv('../data/talks-v1b.csv', names=colnames)
```

Once this is done, you can filter various columns into lists using the `list =  df.column.tolist()` expression: 

```python
talks = df.text.tolist()
authors = df.author.tolist()
dates = df.date.tolist()
```
**N.B.**: *I'm doing it this way because I know how to feed a list of strings into the topic modeling libraries. There may very well be another way to get the date out of the dataframe and into the topic models.*

After this, I pulled out the years using some regex and then zipped the years back with the authors to create what looks like an autho-date citation. I then ran a quick check to make sure things are still in sync:

```python
cited_texts.head()
```

  | citation	          | text
--|-----------------------|------------------------------------------
0 | Al Gore 2006	      | Thank you so much Chris. And it's truly
1 | David Pogue 2006	  | Hello voice mail my old friend. 
2 | Cameron Sinclair 2006 | I'm going to take you on a journey 
3 | Sergey Brin + Larry Page 2007 | Sergey Brin I want to discuss a question
4 | Nathalie Miebach 2011 | What you just heard are the interactions

Now, all the work on determining empty or too short talks get re-called from the file `drop_talks.txt` and loaded into a list. I combined the usual file open into a list comprehenshion in order to convert the numbers being stored as strings into integers. Essentially, I took `numbers = [ int(x) for x in numbers ]` and replaced the `open()` sequence inside the comprehension. And ... it works! (Not sure how Pythonic it is.)

```python
the_bad = [ int(x) for x in open("../data/drop_talks.txt", "r").read().split('\n') ]
```

And then it's time to use KK's backwards technique to keep the indices in place:

```python
for index in sorted(the_bad, reverse=True):
    del talks[index]
```

However, this throws off our pairing, so what we need to do is filter out the rows in the dataframe before creating the lists above. Here's my plan:

* Filter the rows using the index
* Save the new dataframe as a CSV file
* Start with this dataframe as the basis for the work.

First, grab the list of empty or too short talks:

```python
the_bad = [ int(x) for x in open("../data/drop_talks.txt", "r").read().split('\n') ]
```

Second, reverse the order just to be safe:

```python
dab_eht = sorted(the_bad, reverse=True)
```

Now, to purge the dataframe: I need to remove these rows. An example on Quora seems short and to the point:

> In the example below, the rows 1,3,5, and 7 are removed.
> `iris.drop(iris.index[[1,3,5,7]])`

So I'm going to try the following:

```python
df_purged = df.drop(df.index[dab_eht])
```

A quick comparison of the two dataframes: `df[184:187]` versus df_purged[183:186] shows that the operation was successful. (And that the row numbering remains unchanged such that the rows are 184 .. 186 in the new dataframe.)

Saving is simple:

```python
df_purged.to_csv('../data/talks_2.csv')
```

**N.B.**: Apart from the dropped lines, there was another difference between these two files: `talks_2` had the names of the headers written in the first line: `,author,title,date,length,text`. For the sake of simplicity and congruence with the earlier files, I removed the first line (by hand).

Okay, so we need a couple of things going in:

1. We need to use our custom stopword list, `tt_stop.txt`. 
2. We want to use the NLTK tokenizer because it preserves contractions with our terms and so we will need a `for` loop to turn each of the talks as strings into a talk as a list and then understand how to feed that list into the LDA implementation.

So here's a test set of documents, which includes a contraction:

```python
doc_a = "You can call me Al."
doc_b = "I can call you Betty."
doc_c = "Who'll be my role model?"

doc_set = [doc_a, doc_b, doc_c]
```

And here's the loop (uncommented): 


```python
for i in doc_set:
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if not i in stopwords]
    texts.append(stopped_tokens)

print(texts)
[['call', 'al.'], ['call', 'betty.'], ["who'll", 'role', 'model?']]
```

Oops! Other punctuation is showing up. This change to the `raw` line in the loop clears that problem:

```python
raw = re.sub(r"[^\w\d'\s]+",'', i).lower()
```

## 2017-02-04

The working code for the LDA model is, I think, reasonably clear and clean. What it doesn't have built into it, and I have seen other examples that do this, is removal of low-frequency words -- thresholds are variable -- but since most topic models are represented as words that co-occur, I don't know that worrying about the low-freq words is worth the trouble.

A lot of what is at the end of the LDA notebook right now is a variety of attempts to print out the data in the LDA model so that I can also understand how it's stored and how to output it in ways that we can do other things with it. For now, there's a simple for loop that simply makes it easier to copy and paste the output into a CSV -- I know we could write code to do this, but I got lazy at this moment. 

- [] Getting the rich data in the LDA model outputted in a fashion where we can do other things with it is a higher priority.

## 2017-02-07

Okay, with the LDA code working and tested with 25 topics after 10 passes, I ran the code a couple of times with 35 topics and 100 passes -- this took about 5 to 10 minutes on my MacBook Pro, so be prepared to make a cup of coffee.

I checked the 35 LDA topics against those produced by NMF: the NMF looks better. I'm not really sure on the LDA topics at all. I will try it again without the stopwords coming out to see if the results are more in-line with the NMF method.

And the NMF code is a lot faster than the LDA code (at least the `gensim` implementation). 











***
## References

Fredrik deBoer, Evaluating the comparability of two measures of lexical diversity, System, Volume 47, December 2014, Pages 139-145, ISSN 0346-251X, http://dx.doi.org/10.1016/j.system.2014.10.008.


## Notes

[^1]: None of the tokens have been stemmed, so frequency and frequencies, for example, are counted separately. Stemming is not entirely a given in this work. [Frederik DeBoer][] notes in "Evaluating the Comparability of Two Measures of Lexical Diversity" that: "It’s enough to say here that in most computerized attempts to measure lexical diversity, such as the ones I’m discussing here, all constructions that differ by even a single letter are classified as different terms. In part, this is a practical matter, as asking computers to tell the difference between inflectional grammar and derivational grammar is currently not practical. We would hope that any valid measure of lexical diversity would be sufficiently robust to account for the minor variations owing to different forms." On inflectional grammar versus derivational grammar, note: "Inflection is the process of adding inflectional morphemes which modify a verb's tense or a noun's number, rarely affecting the word's meaning or class. Examples of applying inflectional morphemes to words are adding *-s* to the root dog to form dogs and adding *-ed* to wait to form waited. In English, there are eight inflections. In contrast, derivation is the process of adding derivational morphemes, which create a new word from existing words and change either the semantic meaning or part of speech of the affected word, for example by changing a noun to a verb." ([Wikipedia](https://en.wikipedia.org/wiki/Inflection#Inflection_vs._derivation))

[Frederik DeBoer]: http://fredrikdeboer.com/2014/10/13/evaluating-the-comparability-of-two-measures-of-lexical-diversity/