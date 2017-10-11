import csv


reader = csv.reader(open('../data/talks_6c.csv', 'rb'))
writer = csv.writer(open('../data/talks_6d.csv', 'w'))
headers = reader.next()
headers.append("numDate")
writer.write(headers)

MY = {'Jan':'01','Feb':'02','Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 
    'Jul':'07','Aug':'08','Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}

for row in reader:
	mon, year = row['date'].split(" ")
	numd = year + MY[mon]
	row.append(numd)
	writer.write(row)


# based on code found here - 
# https://stackoverflow.com/questions/40229469/add-a-column-to-an-existing-csv-file-with-python

# Other sources consulted 
# https://docs.python.org/3/library/csv.html
# https://www.python.org/dev/peps/pep-0008/#should-a-line-break-before-or-after-a-binary-operator
# https://docs.python.org/3/tutorial/datastructures.html