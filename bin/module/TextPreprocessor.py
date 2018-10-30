from typing import List
import os
import pprint
import bin.setting as setting


#--Tokenize
class Tokenizer():
    """
    Input: Raw article text.
    """
    def __init__(self, article: str):
        self.result = article
        
    def tokenize(self) -> List[List[str]]:
        """
        Output: [[], []].
        """
        self.result += 'test'
        return self
        #(itemId, articleId, [[], []])
        # df['Review_tokenized_sent'] = df['Review'].astype('str').apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
    
    def flatSentences(self) -> List[str]:
        """
        Output: [].
        """
        # df['Review_tokenized_arti'] = df['Review'].astype('str').apply(lambda x: nltk.word_tokenize(re.sub(titleSubStr, '', x)))


#--Normalize
class Normalizer():

    def __init__(self, word: str):
        self.result = word

    def lower(self):
        self.result = self.result.lower()

    def filterNonWord(self):
        self.result = self.result if self.result.isalpha() else str()
    
    def filterStop(self, stopwords=setting.config.TextPreprocessor.stopwords):
        self.result = self.result if self.result not in stopwords else str()

    def keep(self, words2Keep=setting.config.TextPreprocessor.words2Keep):
        self.result = self.result if self.result in words2Keep else str()

    def filter(self, words2Filter=setting.config.TextPreprocessor.words2Filter):
        self.result = self.result if self.result not in words2Filter else str()

    def lemmatize(self, lemmatizer=setting.config.TextPreprocessor.lemmatizer):
        self.result = lemmatizer.lemmatize(self.result)

    def stem(self, stemmer=setting.config.TextPreprocessor.stemmer):
        self.result = stemmer.stem(self.result)

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
        