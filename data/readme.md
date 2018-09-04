# About the Data

Katherine Kinnaird and John Laudun[1](#about-us)

The files in this folder are downloads from two sources, the first of which is the list maintained as a Google Sheet and the second of which are the HTML files downloaded from the TED website as per their usage permissions.

We first became aware of the list when [Open Culture][] reported its existence and that it was being kept up to date. At the time of the report in 2014, the list contained 1756 TED talks. In 2016, when we first decided to begin collaborating and decided to make TED talks a place to start having a dialogue about the application of statistical methods to humanistic topics, the list contained 2209 talks. The list is maintained as a [Google Sheet][], and we have never been able to discern its author(s). In both 2016 and more recently for this version of the data, we exported the list as a CSV file and then parsed and amended the URLs in the CSV as described [below][#from-a-list-to-folders-of-files] in order to create text files that contained lists of URLs that we could feed `wget`.

All of the files we downloaded are here both as an archive of the TED website itself as well as for those cases where others see data within the files that they wish to parse out for themselves. While the latter usage is obvious, we should note that it became apparent as we worked with the HTML files that we had downloaded for this round that the TED website had undergone a significant re-write of its underlying HTML between 2016 and 2018. `BeautifulSoup` code either returned unexpected results or simply returned errors. The code here is from the most recent parsing of the website. We are happy to make the 2016 files available to anyone interested in differences between both the code and the content of the TED website itself. That was not our concern in our current work.

Setting aside the HTML files containing the discussions for the time being, we used Python's `BeautifulSoup` module to parsed the descriptions and the transcripts into two separate CSVs that we then merged into one file in which, we think, most readers will be interested. We edited certain features of the file, correcting for a few parsing errors and also making it easier for the file to load then edited to make it easier to load. As our code reveals, our work is based in Python, and we used its `pandas` library for a lot of the data handling.

What is not included in this particular directory are those CSVs which we think offer extended metadata, but metadata derived from some form of computational analysis. Thus, for example, the CSV which contains the gender(s) for the speaker(s) of a given talk is not in this directory, but in `/outputs`.

What follows is an account of the steps we took to get the list of talks, get the HTML, and then parse the HTML into a CSV.


## From a List to Folders of Files

In both 2016 and 2018, our work began with downloading the list of TED talks maintained as a Google sheet. One thing to note is that this record has changed its structure over time. In May 2016, the columns were labelled as follows:

> 18, id, /, Speaker, Name, Short Summary, Event, Duration, Publish date

By May 2018, that structure had changed to the following:

> Talk ID, public_url, speaker_name, headline, description, event, duration, language, published, tags

Since we were after a complete download of the files on both occasions, we focused our attention on the URLs (titled as "/" in 2016). One of the first things we noticed was that the URLs had changed. In 2016, the URLs were by the ID number -- e.g., http://www.ted.com/talks/view/id/53 -- but in the current moment they are a blend of author and title -- e.g., https://www.ted.com/talks/majora_carter_s_tale_of_urban_renewal. In both cases, this URL takes you to the talk's main page on the TED website, which contains the talk's description and provides all of the material found in the Google sheet.

Because it is easy to use, can be made to access servers respectfully, and accepts a list of inputs as a text file, we used `wget` to download the HTML files from the TED website. (The version of `wget` we used is available through the MacPorts package management system for macos.) We arrived at three lists: the one for the main page which came from the unadulterated URL from the Google sheet, the one for the transcript page which we derived by appending `/transcript` to the end of each of the URLs in the first list, and then the one for the discussions by appending `/discussion` to each URL. The complete command we used for each of the three lists was as follows with each list being issued separately.

```bash
wget -w 2 -i ../descriptions_URL_list.txt
```

Total time to download all the files for each iteration was right around two hours in all three cases. `wget` reported 2749 files for the descriptions and discussions, and 2689 files for the transcripts. (What is not yet transcribed is something we plan to explore as part of the project.)


## Parsing HTML Files into CSVs

While a more detailed account of the work we did is available in the Jupyter notebooks available in this directory, we thought it might be helpful to gloss our actions there for readers interested only in a sketch of the actions that lie behind the data: the notebooks will always be there for fuller consideration and documentation. 

## About Us

Katherine Kinnaird is Clare Boothe Luce Assistant Professor of Computer Science, and Statistical & Data Sciences at Smith College. John Laudun is Doris H. Meriwether/BORSF Endowed Professor of English at the University of Louisiana at Lafayette.

[Open Culture]: http://www.openculture.com/2014/06/1756-ted-talks-listed-in-a-neat-spreadsheet.html
[Google Sheet]: https://docs.google.com/spreadsheets/d/1Yv_9nDl4ocIZR0GXU3OZuBaXxER1blfwR_XHvklPpEM/edit?hl=en&hl=en&hl=en#gid=0
