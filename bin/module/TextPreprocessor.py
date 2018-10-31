import os
import pprint
import nltk
from typing import List
import bin.module.util as util
from bin.setting import path, textPreprocessor as config


#--Decide if keep sentence structure
class Tokentype():
    ARTICLE = 'Tokentype_ARTICLE'
    SENTENCE = 'Tokentype_SENTENCE'


#--Tokenize a list of articles
class Tokenizer():

    def __init__(self, tokenType, articles):
        self.tokenType = tokenType
        self.articles = articles
    
    def tokenize(self):
        if self.tokenType == Tokentype.SENTENCE:
            tokens = ((nltk.word_tokenize(s) for s in nltk.sent_tokenize(a)) for a in self.articles)
        else:
            tokens = (nltk.word_tokenize(a) for a in self.articles)
        return self.tokenType, tokens
    
    @classmethod
    def test_tokenize(cls):
        """
        Print out sample articles, tokens with sentence structure, and tokens without sentence structure
        """
        #2 articles with 2 sentences each
        articles = ['This is a test for text preprocessing. Do you think this could be a good way to expand your knowledge?', 'Is that because theres always an inherent overhead to using classes in Python? And if so, where does the overhead come from technically speaking.']
        print(articles)
        print(util.createListFromGen(cls(Tokentype.SENTENCE, articles).tokenize()[1]))
        print(util.createListFromGen(cls(Tokentype.ARTICLE, articles).tokenize()[1]))


#--Normalize a list of tokens, w/ or w/o sentence structure
class Normalizer():
    """
    Input (1 of the 2):
    - Use tokenType for identification.
    - List of articles, with list of sentences -> [[[words], [words]]].
    - List of articles, with list of words -> [[words], [words]].
    - ((generator A) if sentenceType else (generator B) for article in result)
    """

    def __init__(self, tokenType, tokens):
        self.tokenTypeSentence = tokenType == Tokentype.SENTENCE
        self.result = tokens
    
    def lower(self):
        self.result = (((tk.lower() for tk in st) for st in a) if self.tokenTypeSentence else (tk.lower() for tk in a) for a in self.result)
        return self

    def filterNonWord(self):
        self.result = (((tk for tk in st if tk.isalpha()) for st in a) if self.tokenTypeSentence else (tk for tk in a if tk.isalpha()) for a in self.result)
        return self
    
    def filterStop(self, stopwords=config.stopwords):
        self.result = (((tk for tk in st if tk not in stopwords) for st in a) if self.tokenTypeSentence else (tk for tk in a if tk not in stopwords) for a in self.result)
        return self

    def keep(self, words2Keep=config.words2Keep):
        self.result = (((tk for tk in st if tk in words2Keep) for st in a) if self.tokenTypeSentence else (tk for tk in a if tk in words2Keep) for a in self.result)
        return self

    def filter(self, words2Filter=config.words2Filter):
        self.result = (((tk for tk in st if tk not in words2Filter) for st in a) if self.tokenTypeSentence else (tk for tk in a if tk not in words2Filter) for a in self.result)
        return self

    def lemmatize(self, lemmatizer=config.lemmatizer):
        self.result = (((lemmatizer(tk) for tk in st) for st in a) if self.tokenTypeSentence else (lemmatizer(tk) for tk in a) for a in self.result)
        return self

    def stem(self, stemmer=config.stemmer):
        self.result = (((stemmer(tk) for tk in st) for st in a) if self.tokenTypeSentence else (stemmer(tk) for tk in a) for a in self.result)
        return self
    
    @classmethod
    def test_operations(cls):
        """
        Test chained lower(), filterStop(), stem()
        """
        tokens_sentence = [[['test', 'Is', 'gooD', '.'], ['HELLO', 'world']], [['overhead', 'comes', 'from', 'technically', 'speaking']]]
        tokens_article = [['test', 'Is', 'gooD', '.', 'HELLO', 'world'], ['overhead', 'comes', 'from', 'technically', 'speaking']]
        print('Test w/ sentence structure:')
        print(tokens_sentence)
        print(util.createListFromGen(cls(Tokentype.SENTENCE, tokens_sentence).lower().filterStop().stem().result))
        print('Test w/o sentence structure:')
        print(tokens_article)
        print(util.createListFromGen(cls(Tokentype.ARTICLE, tokens_article).lower().filterStop().stem().result))


#--A df row generator
#Parameters: chunkSize and noOfChunk both int
#StartRow should be a multiple of chunkSize
#Read in by chunk (save disk access times) but yield by row
#Use next to get
class DfDispatcher():

    def __init__(self, filePath, chunkSize, startRow):
        import pandas as pd

        self.dfIter = pd.read_csv(filePath, chunksize=chunkSize)
        self.startRow = startRow
        self.curRow = 0
        self.chunkSize = chunkSize

        i = 0
        while i <= startRow / chunkSize:
            next(self.dfIter)
    
    def __iter__(self):
        self.curRow += self.chunkSize

        for chunk in self.dfIter:
            yield from chunk.iterrows()


#--Average no of sentence/no of words
class Briefer():
    #Word frequency distribution by nltk
    fdist = FreqDist([i for i in flatten_list(text_preprocessed)])

    #Observe result
    print('Unique terms:', fdist.B())
    print('Total terms:', fdist.N())
    sorted(fdist.items(), key=operator.itemgetter(1), reverse=True) #Top terms
        