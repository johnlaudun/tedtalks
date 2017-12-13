import csv

# Open the input file to get the headers more easily 
reader = csv.reader(open('../data/talks_6c.csv'))
headers = reader.__next__()
headers.append("numDate")

ind = headers.index('date')


# Dictionary that coverts the shortened months into numbers:
MY = {'Jan':'01','Feb':'02','Mar':'03', 'Apr':'04', 'May':'05', 
    'Jun':'06', 'Jul':'07','Aug':'08','Sep':'09', 'Oct':'10', 
    'Nov':'11', 'Dec':'12'}

with open('../data/talks_6d.csv', 'w') as csvfile:
	# Make the output file. 
	writer = csv.writer(csvfile)
	writer.writerow(headers)

	# Copy each row from the input file and add date in YYYYMM format
	for row in reader:
		mon, year = row[ind].split(" ")
		numd = year + MY[mon]
		row.append(numd)
		writer.writerow(row)


# based on code found here - 
# https://stackoverflow.com/questions/40229469/add-a-column-to-an-existing-csv-file-with-python

# Other sources consulted 
# https://docs.python.org/3/library/csv.html
# https://www.python.org/dev/peps/pep-0008/#should-a-line-break-before-or-after-a-binary-operator
# https://docs.python.org/3/tutorial/datastructures.html