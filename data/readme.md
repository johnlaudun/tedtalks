# About the Data

Katherine Kinnaird and John Laudun[1](#about-us)

The files in this folder are downloads from two sources, the first of which is the list maintained as a Google Sheet and the second of which are the HTML files downloaded from the TED website as per their usage permissions.

We first became aware of the list when [Open Culture][] reported its existence and that it was being kept up to date. At the time of the report in 2014, the list contained 1756 TED talks. In 2016, when we first decided to begin collaborating and decided to make TED talks a place to start having a dialogue about the application of statistical methods to humanistic topics, the list contained 2209 talks. The list is maintained as a [Google Sheet][], and we have never been able to discern its author(s). In both 2016 and more recently for this version of the data, we exported the list as a CSV file and then parsed and amended the URLs in the CSV as described [below][#from-a-list-to-folders-of-files] in order to create text files that contained lists of URLs that we could feed `wget`.

All of the files we downloaded are here as an archive, both of the TED website itself as well as for those cases where others see data within the files that they wish to parse out for themselves. While the latter is obvious, we should note that it became apparent as we worked with the HTML files ourselves that we had downloaded for this round that the TED website had undergone a significant re-write of its underlying HTML between 2016 and 2018. `BeautifulSoup` code either returned unexpected results or simply returned errors. The code here is from the most recent parsing of the website. We are happy to make the 2016 files available to anyone interested in differences between both the code and the content of the TED website itself. That was not our concern in our current work.

Using Python's `BeautifulSoup` module, we parsed the descriptions and the transcripts into 
Most readers will be interested, we think, in the CSV file which represents the data parsed from the various HTML files and then edited to make it easier to load: as our code reveals, we ultimately based our work on Python and used its `pandas` library for a lot of the data handling.

What is not included in this directory are those CSVs which we think offer extended metadata, but metadata derived from some form of computational analysis. Thus, for example, the CSV which contains the gender(s) for the speaker(s) of a given talk is not in this directory, but in `/outputs`.

What follows is an account of the steps we took to get the list of talks, get the HTML, and then parse the HTML into a CSV.


## From a List to Folders of Files








## About Us

Katherine Kinnaird is Clare Boothe Luce Assistant Professor of Computer Science, and Statistical & Data Sciences at Smith College. John Laudun is Doris H. Meriwether/BORSF Endowed Professor of English at the University of Louisiana at Lafayette.

[Open Culture]: http://www.openculture.com/2014/06/1756-ted-talks-listed-in-a-neat-spreadsheet.html
[Google Sheet]: https://docs.google.com/spreadsheets/d/1Yv_9nDl4ocIZR0GXU3OZuBaXxER1blfwR_XHvklPpEM/edit?hl=en&hl=en&hl=en#gid=0
