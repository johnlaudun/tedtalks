# For loop for stemming talks as needed 
stemmed = []
for talk in talks:
    raw = re.sub(r"[^\w\d'\s]+",'', i).lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if not i in stopwords]
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    stemmed.append(stemmed_tokens)

unstemmed_words = set([y for x in unstemmed for y in x])
stemmed_words = set([y for x in stemmed for y in x])


# Build master list of words:
unstemmeds = [y for x in unstemmed for y in x]
stemmeds = [y for x in stemmed for y in x]

# Create dictionary of word:frequency pairs

punctuation = re.compile(r'[.?!,":;]') 
stemmed_freq_dict = {}

for word in stemmeds:
    # remove punctuation marks
    word = punctuation.sub("", word)
    # form dictionary
    try: 
        stemmed_freq_dict[word] += 1
    except: 
        stemmed_freq_dict[word] = 1
        
stemmed_freq_list = [(val, key) for key, val in stemmed_freq_dict.items()]
stemmed_word_list = [(key, val) for key, val in stemmed_freq_dict.items()]

stemmed_word_list.sort()
print(stemmed_word_list)