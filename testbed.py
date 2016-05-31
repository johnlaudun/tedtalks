import re
import csv
import os
from bs4 import BeautifulSoup

def parse(the_soup):
    # both title and author are can be parsed in separate tags.
    author = the_soup.select_one("h4.h12.talk-link__speaker").text
    title = the_soup.select_one("h4.h9.m5").text
    # just need to strip the text from the date string, no regex needed.
    date = the_soup.select_one("span.meta__val").text.strip()      
    # we want the last time which is the talk-transcript__para__time previous to the footer.
    mn, sec = map(int, the_soup.select_one("footer.footer").find_previous("data", {
    "class": "talk-transcript__para__time"}).text.split(":"))
    length = (mn * 60 + sec)        
    # to ignore (Applause) etc.. we can just pull from the actual text fragment checking for (
    text = " ".join(d.text for d in the_soup.select("span.talk-transcript__fragment") if not d.text.startswith("("))        
    # clean the text
    text = re.sub('[^a-zA-Z\.\']', ' ', text)
    return  author.strip(), title.strip(), date, length, text


def to_csv(pth, out):
    # open file to write to.
    with open(out, "w") as out:
        # create csv.writer. 
        wr = csv.writer(out)
        # write our headers.
        wr.writerow(["author", "title", "date", "length", "text"])
        # get all our html files.
        for html in os.listdir(pth):
            with open(os.path.join(pth, html)) as f:
                # parse the file are write the data to a row.
                wr.writerow(parse(BeautifulSoup(f, "lxml")))
				
to_csv("./test2","test2.csv")