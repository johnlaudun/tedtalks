# Notebook

## Getting the transcripts

The next step was how to download all the transcripts. The URLs all looked like this:

    https://www.ted.com/talks/stuart_firestein_the_pursuit_of_ignorance/transcript?language=en

wget is flexible, but not so flexible that you could fead it a variable to fill in everything between `talks` and `transcript`. However, if you feed it a list of files, it will work its way through that list. A search of the web turned up a post on [Open Culture][] describing a list of the 1756 TED Talks available in 2014. As luck would have it, the [Google Spreadsheet][] is still being maintained.

I downloaded the spreadsheet as a CSV file and then simply grabbed the column of URLs using Numbers. (This could have been done with `pandas` but it would have taken more time, and I didn’t need to automate this part of the process.) The URLs were to the main page for each talk, and not the transcript, but all I needed to do was to add the following to the end of each line:

    /transcript?language=en

Which I did with some of the laziest regex ever. I could then cd into the directory I created for the files and ran this:

    wget -w 2 -i ~/Desktop/talk_list.txt

What remains now is to use Beautiful Soup to rename the files using the html title tag and to get rid of everything but the actual transcript. Final report from wget:

    FINISHED --2016-05-18 16:16:52--
    Total wall clock time: 2h 14m 51s
    Downloaded: 2114 files, 153M in 3m 33s (735 KB/s)


## From HTML to CSV

### Author
```
<h4 class='h12 talk-link__speaker'>Al Gore</h4>
```

### Author and Title
```
<title>Al Gore: Averting the climate crisis | TED Talk Subtitles and Transcript | TED.com</title>
```

### Date
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

### Length

Length is probably best estimated by the last of the times:
```
<data class='talk-transcript__para__time'>
15:57
</data>
```

### Text

It looks like the actual "talk" is contained in the the div: `<div class='talk-article__body talk-transcript__body'>` and then paragraphing is achieved with: `<span class='talk-transcript__para__text'>`. (It would be nice, perhaps, to keep the paragraphing? But it is, above all, a transcription artifact.)


## CSV

I think I want all this in a CSV in order to have a data structure. If so, what I want to do: 

* Pull author either from `h4` or from `title` and place it in an *author* column.
* Pull title from `title` and place it in a *title* column.
* Pull date from `meta` and place it in *date* column.
* Pull the length of the file from the transcript above and place it in a *length* column.
* Pull the text from `talk-transcript__body` and place it in a *text* column.

I think I can do all this with **`BeautifulSoup`**, and I imagine I need to:

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