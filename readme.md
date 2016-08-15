# Ten Years of TED Talks

**Table of Contents**

1. [Reasons for Interest](#reasons-for-interest)
2. [Features under Consideration](#features-under-consideration)
3. [CFPs](#cfps)
4. [Getting the Transcripts](#getting-the-transcripts)
5. [Generating a CSV](#generating-a-csv)

## Reasons for Interest

* incredible saturation of various dimensions of culture very quickly, both popular and on NPR.
* impact on education (TEDucation)
* A medium form: we see repeatedly the gap between the study of culture as tweets and the study of culture as novels. There is an incredible middle ground and these talks are representative of that.


## Features Under Consideration

### Features Previously Discussed

* This initial feature set is part of Sebastian Wernicke’s TED talk (http://www.ted.com/talks/lies_damned_lies_and_statistics_about_tedtalks), but his version of events is not alone in trying to provide a framework or formula for giving great talks or speeches. I also want to see if there are features to be gleaned from talks listed below:
  - June Cohen: What Makes a Great TEDTalk (https://youtu.be/RVDfWfUSBIM)
  - Nancy Duarte: The secret structure of great talks (https://youtu.be/1nYFpuc2Umk)
  - How to TEDx: How to give a great TEDx Talk (https://youtu.be/6H67JgLwyF4)
  - Will Stephen has a funny take on all this (https://youtu.be/8S0FDjFBj8o)
* Wernicke’s claims
  - 1 week of video, 1 million words, 2 million ratings
  - Goal: the ultimate TED talk and the worst possible (but still acceptable)
  - Foci: topics used, style of delivery, and visuals used.
  - Sorted talks in two dimensions: ideas/actions, emotional/rational.
  - Top ten words in the most favorite TED talks and the least favorite. > Then the topics that TED audiences like and those they do not like.
  - Also compared lengths (in time) of the most and least favorited talks. (Included comments in this because he was able to report that shorter talks got comments like “beautiful.”
  - 4-grams for most and least favorite.
  - He has a few more things to consider (e.g., the “TED Pad”).
    

### Possible Features to Consider

* What is the overall “topical map” of the TED talks and what is changing map of the talks: i.e., year to year.
* Do topics arise in response to external events?
* What is the most central TED talk once we have a set of features?
* Do the early talks set any kind of example or reveal any kind of influence in later talks in terms of topics or word use?
* Are their consistent phrases? (See Wernicke above.)
* Do these talks reveal any particular shape? (In terms of sentiment or other controlled-vocabulary methods.)
* Speaking of which, is there a TED vocabulary/lexicon?
* How do any of these features play out against how often a talk is viewed/favorited? (We have another measure of sentiment, which means we also have a way to consider how the sentiment about a talk compares to sentiment within a talk.)
* Time 100 vs TED speakers. Which comes first? Who comes across? >>> Time's list is published in April.
* Sentiment vs Public Mood (Stimson).
* Some TEDs have tag lines: do talks follow the "assigned topic"?
* When does a TED talk “deliver” its title? (Train topics on documents and then hand code titles.)
* Dynamic Topic Modeling: is there influence? (Probably not but if null, interesting?)
* ted.com/speakers has all the speakers and it has them by some sort of category/designation. >>> Grab speaker profiles from TED website, build CSV with name, designation, and text. (We will derive gender from text.)
* Text length/time would give us a sense of pacing. And if we can pull the popularity of the talks into the analysis, we might know something about what's the "preferred" pace for TED talks.
    - Talk lengths: by class.
    - Density of terms vs. density of talk. TED talks are more dense in what way? (Less framing language, metalanguage, metadiscourse? >>> This will probably be phrases.)


### More Features

* Organization/formulae/templates for talks:
  - Sampled talk had a quick introduction and a quick definition and then lots of details: do technical words increase in the "body" of the talk? Second sample also used "I hope" at the end of the talk.
  - Derived "footprint" by audio analysis? (Footprint = structure = organization.)
  - Last sentence (or so) of every talk.
  - If TED has "thin" introductions and conclusions and seem to focus more on examples within a talk, are there fewer "structural" words or cues? (e.g., "in this next section" "I will now examine.")
  - In the big "middle" of TED talks, are there signals (phrases or audience applause) that indicate a better talk? (A more organized talk?)
* James' Short List
  - Footprint
  - External signals
  - Stock words/phrases
  - Density 


## CFPs

### EKAW 2016

The following CFP came across _The Humanist_ mailing list on 2016 May 26. Top portion quoted below for the notion of *trend detection*:


>CALL FOR PAPERS
>20th International Conference on Knowledge Engineering and Knowledge
>Management (EKAW 2016)
>
>19-23 November 2016, Bologna, Italy
>Abstract submission: July 8, 2016
>Paper submission: July 15, 2016
>Web site: http://ekaw2016.cs.unibo.it/?q=callforpapers
>
>The 20th International Conference on Knowledge Engineering and Knowledge
>Management is concerned with the impact of time and space on the
>representation of knowledge. Knowledge engineering has mostly been about
>creating static, universal representations. Yet the world is rarely static:
>everything changes, including the models, and real world systems need to
>evolve along with the surrounding world. Also, what makes some
>representations valid in some contexts may make them invalid elsewhere
>(e.g., jurisdiction for laws).
>
>The special focus of this year's EKAW is "evolving knowledge", which
>concerns all aspects of the management and acquisition of knowledge
>representations of evolving, contextual, and local models. This includes
>change management, trend detection, model evolution, streaming data and
>stream reasoning, event processing, time-and space dependent models,
>contextual and local knowledge representations, etc.
>
>EKAW 2016 will put a special emphasis on the evolvability and localization
>of knowledge and the correct usage of these limits.


## Getting the transcripts

The first step was how to download all the transcripts. The URLs all looked like this:

    https://www.ted.com/talks/stuart_firestein_the_pursuit_of_ignorance/transcript?language=en

`wget` is flexible, but not so flexible that you could feed it a variable to fill in everything between `talks` and `transcript`. However, if you feed it a list of files, it will work its way through that list. A search of the web turned up a post on [Open Culture][] describing a list of the 1756 TED Talks available in 2014. As luck would have it, the [Google Spreadsheet][] is still being maintained.

I downloaded the spreadsheet as a CSV file and then simply grabbed the column of URLs using Numbers. (This could have been done with `pandas` but it would have taken more time, and I didn’t need to automate this part of the process.) The URLs were to the main page for each talk, and not the transcript, but all I needed to do was to add the following to the end of each line:

    /transcript?language=en

Which I did with some of the laziest regex ever. I could then cd into the directory I created for the files and ran this:

    wget -w 2 -i ~/Desktop/talk_list.txt

What remains now is to use Beautiful Soup to rename the files using the html title tag and to get rid of everything but the actual transcript. Final report from wget:

    FINISHED --2016-05-18 16:16:52--
    Total wall clock time: 2h 14m 51s
    Downloaded: 2114 files, 153M in 3m 33s (735 KB/s)


### From HTML to CSV

#### Author
```
<h4 class='h12 talk-link__speaker'>Al Gore</h4>
```

#### Author and Title
```
<title>Al Gore: Averting the climate crisis | TED Talk Subtitles and Transcript | TED.com</title>
```

#### Date
```
<div class='meta'>
<span class='meta__item'>
Posted
<span class='meta__val'>
Jun 2006
</span>
</span>
<span class='meta__row'>
Rated
<span class='meta__val'>
Funny, Informative
</span>
</span>
</div>
```

#### Length

Length is probably best estimated by the last of the times:
```
<data class='talk-transcript__para__time'>
15:57
</data>
```

#### Text

It looks like the actual "talk" is contained in the the div: `<div class='talk-article__body talk-transcript__body'>` and then paragraphing is achieved with: `<span class='talk-transcript__para__text'>`. (It would be nice, perhaps, to keep the paragraphing? But it is, above all, a transcription artifact.)


## Generating a CSV

Next we assembled everything into a CSV in order to have a better sense of our data structure as well as a common document from which we could all draw: 

* Pull *author* either from `h4` or from `title` and place it in an *author* column.
* Pull *title* from `title` and place it in a *title* column.
* Pull *date* from `meta` and place it in *date* column.
* Pull the *length* of the file from the transcript above and place it in a *length* column.
* Pull the *text* from `talk-transcript__body` and place it in a *text* column.

We did all this with **`BeautifulSoup`** which is able to:

1. Read a file.
2. Grab these elements and place them in a list.
3. Write list to line in CSV file.


## Rename Files

I'm looking for this:

   <link href="http://www.ted.com/talks/al_gore_on_averting_climate_crisis/transcript" rel="canonical" />

And I want to name the file: al_gore_on_averting_climate_crisis.

   # Early attempt at re-naming files
   filepath = os.path.abspath('/Users/john/Code/ted/transcript?language=en')
   fileopen = open(filepath).read()
   title = soup.find_all('link', 'rel="canonical"')

   # os.rename(old_file_name, new_file_name)
   print(title)

   # Early attempt to get the date:
   #date_spans = soup.find_all('span', {'class' : 'meta__val'})
   #date = [x.get_text().strip("\n\r") for x in date_spans if re.search(r"(?s)[A-Z][a-z]{2}\s+\d{4}", x.get_text().strip("\n\r"))][0]

[Open Culture]: http://www.openculture.com/2014/06/1756-ted-talks-listed-in-a-neat-spreadsheet.html
[Google Spreadsheet]: https://spreadsheets.google.com/ccc?hl=en&key=pjGlYH-8AK8ffDa6o2bYlXg&hl=en#gid=0


## Further On...

LDA and possible visualization: http://chdoig.github.io/pygotham-topic-modeling/#/


## So Much `wget`ting

```bash
FINISHED --2016-06-02 15:48:26--
Total wall clock time: 2h 17m 7s
Downloaded: 1397 files, 116M in 18m 4s (109 KB/s)
```