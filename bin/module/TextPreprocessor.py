from typing import List

#--Tokenize
class Tokenizer():
    """
    Input: Raw article text.
    """
    def __init__(self, article: str):
        self.output = article
        
    def tokenize(self) -> List[str]:
        """
        Output: [[], []].
        """
        self.output += 'test'
        return self
        #(itemId, articleId, [[], []])
        # df['Review_tokenized_sent'] = df['Review'].astype('str').apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
    
    def flatSentences(self):
        """
        Output: [].
        """
        # df['Review_tokenized_arti'] = df['Review'].astype('str').apply(lambda x: nltk.word_tokenize(re.sub(titleSubStr, '', x)))

tokenizer = Tokenizer('hello, good')
test = tokenizer.tokenize().tokenize().output

#--Normalize
class Normalizer():

    def lower(): pass

    def filterNonWord():
    
    def filterStop(): pass

    def keep(): pass

    def filter(): pass

    def lemmatize(): pass

    def stem(): pass

        #Lowering the case and removing non-words
        workingIter = (w.lower() for w in tokenLst if w.isalpha())

        #Lemmertize
        if lemmer is not None:
            workingIter = (lemmer.lemmatize(w) for w in workingIter)

        #Stem
        if stemmer is not None:
            workingIter = (stemmer.stem(w) for w in workingIter)
        
        #Remove unwanted words by reg
        if Xvocab is not None:
            Xvocab_str = '|'.join(Xvocab)
            workingIter = (w for w in workingIter if not re.match(Xvocab_str, w))

        #Include ONLY wanted words by reg
        if Ovocab is not None:
            Ovocab_str = '|'.join(np.array(Ovocab))
            workingIter = (w for w in workingIter if re.fullmatch(Ovocab_str, w))

        #Remove stopwords
        if stopwordLst is not None:
            workingIter = (w for w in workingIter if w not in stopwordLst)
        return list(workingIter)

    def test(): pass

Normalizer(x).filterStop().filter().keep().lemmatize().stem()


#--A df_review row generator
#Parameters: chunkSize and maxChunk both int
#The actual num of rows will be generated = chunkSize * maxChunk
class DfDispatcher():
    def __init__(self, filePath, chunkSize, maxChunk):
        self.df = pd.read_csv(filePath, chunksize=chunkSize)
        self.maxChunk = maxChunk
        self.curChunk = 0
    def __iter__(self):
        for p in self.df:
            if self.curChunk == self.maxChunk: break
            print('Start processing chunk', self.curChunk)
            self.curChunk += 1
            yield from p.iterrows()    


#--Average no of sentence/no of words
class Briefer():
    #Word frequency distribution by nltk
    fdist = FreqDist([i for i in flatten_list(text_preprocessed)])

    #Observe result
    print('Unique terms:', fdist.B())
    print('Total terms:', fdist.N())
    sorted(fdist.items(), key=operator.itemgetter(1), reverse=True) #Top terms
        