'''
Haven't gotten to this section yet...

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# Function for printing topic words (used later):
def print_top_words(model, feature_names, n_top_words):
    for topic_id, topic in enumerate(model.components_):
        print('\nTopic {}:'.format(int(topic_id)))
        print(''.join([feature_names[i] + ' ' + str(round(topic[i], 2))
              +', ' for i in topic.argsort()[:-n_top_words - 1:-1]]))

print_top_words(nmf, tfidf_feature_names, n_top_words) #n_top_words can be changed on the fly

'''

doctopic = nmf.fit_transform(dtm) # This is an array 

# =-=-=-=-=-= 
# Creating arrays in the order that we want them
# =-=-=-=-=-= 

# The AUTHORS, DATES, and CITATIONS variables are lists. We need them to be arrays. 
authors = np.array([authors])  # Note: the seemingly extra [] make the array have 
dates = np.array([dates])      #       dimensions of (k,1) instead of just (k,)
citations = np.array([citations])

# Below sytax is from: http://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list

# We want to sort everything by the dates. To do this, we will use ARGSORT() to 
# give us the indices by which to sort all of our other arrays:

##JL: This is like magic to my mind:

inds = np.argsort(dates)

# Now we will use INDS to sort each of our arrays: DATES, AUTHORS, and DOCTOPIC:
dates = dates[inds]
authors = authors[inds]
doctopic = doctopic[inds]


# =-=-=-=-=-= 
# Saving output to CSV
# =-=-=-=-=-= 

# Since DOCTOPIC is an array, you can just do: 
#      np.savetxt("foo.csv", doctopic, delimiter=",", fmt = "%s")
# http://stackoverflow.com/questions/6081008/dump-a-numpy-array-into-a-csv-file
# 
# The above won't give you the names of the files. Instead try this:

topsnum = np.array([list(range(n_topics))])
# topsnum = np.indices((1,n_topics))[1] <-- this is more than we need, 
#                                           but it's cool to know more tricks
#
# Two ways to get an array that is of the form [[0,1,2,3,...]].
# It will have the desired dimensions of (1,35) which is what we want


fileheader = np.concatenate((np.array([["citations"]]), topsnum),axis = 1) 

docTopics = np.concatenate((citatations, doctopic), axis = 1)
docTopics = np.concatenate((fileheader, docTopics), axis = 0)

np.savetxt("sortedtalks.csv", docTopics, delimiter=",", fmt = "%s")

# =-=-=-=-=-= 
# Finding where to cut the data to get one dataset per year
# =-=-=-=-=-= 

# Here we create a smaller fileheader that will only include the authors. 

yfileheader = np.concatenate((np.array([["authors"]]), topsnum),axis = 1) 

minyear = dates[0]
maxyear = dates[-1]

# We initialize STARTINDS at 0 because our dates are already sorted.
startinds = 0

for year in range(minyear, maxyear):
    lastinds = np.searchsorted(dates, year + 1, "left")
    # The SEARCHSORTED command looks for values in a sorted list. 
    # The "LEFT" flag just gives us the index for the first instance.

    # Slice the rows that we need for this YEAR.
    yearDT = doctopic[startinds:lastinds,:]
    yearA = authors[startinds:lastinds]

    # Add the pieces back together 
    yearDTwA = np.concatenate((yearA,yearDT), axis = 1)
    yearDTwApH = np.concatenate((yfileheader, yearDTwA), axis = 0)

    # Create FILENAME and save the data 
    filename = "talks_year" + str(year) + ".csv"
    np.savetxt(filename, yearDTwApH, delimiter=",", fmt = "%s")



# =-=-=-=-=-= 
# Understanding why and how the printing of the topics works. 
# =-=-=-=-=-= 

for i in range(len(doctopic)): #march over each row --> document
    top_topics = np.argsort(doctopic[i,:])[::-1][0:3] 
    # DOCTOPIC[I,:] is the amount of each topic starting at 0
    # NP.ARGSORT(DOCTOPIC[I,:]) tells us how topics are used starting with the least 
    #        EX - test = np.matrix("1 3 4 2")
    #        	  np.argsort(test) will yield matrix([[0, 3, 1, 2]])
    # [::-1] reverses the order. 
    # So NP.ARGSORT(DOCTOPIC[I,:])[::-1] gives us the topics in order of being used
    #        biggest to smallest
    # Then NP.ARGSORT(DOCTOPIC[I,:])[::-1][0:3] gives us the top 3 topics
    top_topics_str = ' '.join(str(t) for t in top_topics)
    # Here you loop over each of the top three topics and convert the integer type to a 
    # string and concatenates all of these together. 
    print("{}: {}".format(citations[i], top_topics_str))
    # Then you print two strings for each document

"""
Ok so we don't need this, but I already wrote it before I figured that out. 
So I'm leaving it in...

# Pre-allocate the YEARS variable 
years = np.array([np.zeros(n_samples)]).T

for i in range(len(doctopic)): #march over each row --> document
    years[i] = citations[i].split()[-1]
    # CITATIONS is a list of strings
    # .SPLIT() splits the strings at the space. 
    # [-1] takes the last element in the list
"""
