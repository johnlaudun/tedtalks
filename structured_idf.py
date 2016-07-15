term_docs = dict()
ndocs = 0
for fn in os.listdir(DATADIR):
    f = open(os.path.join(DATADIR,fn),'rb')
    text = f.read()
    f.close()
    for sent in nltk.sent_tokenize(text):
        for term in nltk.word_tokenize(sent):
            if term_docs.has_key(term):
                term_docs[term].add(ndoc)
            else:
                term_docs[term].add([ndoc])
    ndocs += 1
idfs = []
for term in term_docs:
    idf = math.log(ndocs/len(term_docs[term]+1)
    idfs.append((term,idf))
print sorted(idfs, key=itemgetter(1), reverse=True)