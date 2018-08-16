# About the Data

The files in this folder are downloads from two sources, the first of which is the the list maintained by as a Google Sheet and the second of which are the HTML files downloaded from the TED website as per their usage permissions. Those files are mostly here as an archive, both of the TED website itself as well in case others see data within the files that they wish to parse out for themselves. Most readers will be interested, we think, in the CSV file which represents the data parsed from the various HTML files and then edited to make it easier to load: as our code reveals, we ultimately based our work on Python and used its pandas library for a lot of the data handling.

What is not included in this directory are those CSVs which we think offer extended metadata, but metadata derived from some form of computational analysis. Thus, for example, the CSV which contains the gender(s) for the speaker(s) of a given talk is not in this directory, but in `/outputs`.

What follows is an account of the steps we took to get the list of talks, get the HTML, and then parse the HTML into a CSV.


## From List to Folders of Files
