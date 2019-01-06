import os
import operator
import nltk
import gensim
import pandas as pd
import numpy as np
from typing import List
from nltk.probability import FreqDist

import bin.module.util as util
from bin.setting import path, textPreprocessor as config

logger = util.initLogger(loggerName='TextPreprocessor')




#TODO: pad sequence
#--Tokenize a list (generator) of articles with sentence structure
class Tokenizer():
    #2 articles with 2 sentences each
    sampleAt = ['This is a test for text preprocessing. Do you think this could be a good way to expand your knowledge?', 'Is that because theres always an inherent overhead to using classes in Python? And if so, where does the overhead come from technically speaking.']
    
    @staticmethod
    def tokenize(articles):
        result = ((nltk.word_tokenize(st) for st in nltk.sent_tokenize(at)) for at in articles)
        logger.info('Tokenized articles.')
        return result
    
    @classmethod
    def brief(cls, articles):
        """
        Input: tokens with sentence structure.
        """
        tokens = util.createListFromGen(cls.tokenize(articles))

        #Word frequency distribution by nltk
        fdist = FreqDist((tk for at in tokens for st in at for tk in st))

        nArticle = len(tokens)
        nSentence = sum(map(lambda article: len(article), tokens))
        nWord = fdist.N()

        logger.info('About the corpus')
        logger.info('- Number of articles: ' + str(nArticle))
        logger.info('- Number of sentences: ' + str(nSentence))
        logger.info('- Number of terms: ' + str(nWord))
        logger.info('- Number of unique terms: ' + str(fdist.B()))
        logger.info('- Top terms:')
        logger.info(sorted(fdist.items(), key=operator.itemgetter(1), reverse=True)[0:5])
        logger.info('- Terms per sentence: ' + str(nWord / nSentence))
        logger.info('- Terms per article: ' + str(nWord / nArticle))
        logger.info('- Sentences per article: ' + str(nSentence / nArticle))
    
    @classmethod
    def test_tokenize(cls):
        """
        Print out sample articles and tokens with sentence structure
        """
        print(cls.sampleAt)
        print(util.createListFromGen(cls.tokenize(cls.sampleAt)))
    
    @classmethod
    def test_brief(cls):
        cls.brief(cls.sampleAt)




#--Normalize a list of tokens, w/ or w/o sentence structure
class Normalizer():
    """
    Input: List of articles, with list of sentences -> [[[words], [words]]].
    """

    def __init__(self, tokens):
        self.result = tokens
    
    def getResult(self):
        logger.info('Normalized {} articles.'.format(len(self.result)))
        return self.result
    
    def lower(self):
        self.result = ((tk.lower() for tk in st) for st in at)
        return self

    def filterNonWord(self):
        self.result = ((tk for tk in st if tk.isalpha()) for st in at)
        return self
    
    def filterStop(self, stopwords=config.stopwords):
        self.result = ((tk for tk in st if tk not in stopwords) for st in at)
        return self

    def keep(self, words2Keep=config.words2Keep):
        self.result = ((tk for tk in st if tk in words2Keep) for st in at)
        return self

    def filter(self, words2Filter=config.words2Filter):
        self.result = ((tk for tk in st if tk not in words2Filter) for st in ta)
        return self

    def lemmatize(self, lemmatizer=config.lemmatizer):
        self.result = ((lemmatizer(tk) for tk in st) for st in at)
        return self

    def stem(self, stemmer=config.stemmer):
        self.result = ((stemmer(tk) for tk in st) for st in at)
        return self
    
    @classmethod
    def test_operations(cls):
        """
        Test chained lower(), filterStop(), stem()
        """
        tokens = [[['test', 'Is', 'gooD', '.'], ['HELLO', 'world']], [['overhead', 'comes', 'from', 'technically', 'speaking']]]
        print(tokens)
        print(util.createListFromGen(cls(tokens).lower().filterStop().stem().getResult()))




#--A df row generator
class DfDispatcher():
    """
    Read in by chunk (save disk access times) but yield by row
    - `chunkSize` = how many rows to read per access.
    - Dispatch between `startRow` and `endRow` (inclusive).
    - Return (rowId, rowContent) for each row.
    """

    def __init__(self, filePath, startRow=0, endRow=None, chunkSize=1000):
        #"cp1252", "ISO-8859-1", "utf-8"
        self.readCsvParam = {
            'filepath_or_buffer': filePath,
            'encoding': 'cp1252',
            'chunksize': chunkSize,
            'nrows': 1 + endRow if endRow else None
        }
        self.startRow = startRow
        self.dfIter = self.__iter__()

        logger.info('Initiated df dispatcher of \"{}\" from row {} to row {}.'.format(filePath, startRow, endRow or 'file ends'))

    def __iter__(self):
        dfIter = (row for chunk in pd.read_csv(**self.readCsvParam) for row in chunk.iterrows())
        i = 0
        #TODO: try use send() instead
        while i < self.startRow:
            i += 1
            next(dfIter)
        return dfIter

    def __next__(self):
        return next(self.dfIter)
    
    def getCol(self, colName):
        return (row[colName] for i, row in self)




#--Load experience keywords and embedding operations
class EmbOperator():
    
    @staticmethod
    def loadPretrainedEmb8Keywords(path):
        """
        Load pretrained emb at `path` and exp keywords
        """
        emb = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        logger.info('Loaded pretrained embedding with', len(emb.index2word), 'words.')

        keyWords = pd.read_csv(path.expKeyword, encoding='utf-8', header=None)[0].tolist()
        keyWords = [word.lower() for word in keyWords]
        logger.info('Load exp keywords of', len(keyWords), 'words.')

        return (emb, keywords)

    @staticmethod
    def getSentenceEmb(at, emb):
        """
        Average emb by sentence.
        - Return: [sent=array(300,)]
        """
        sentenceEmb = [np.array([emb[tk] for tk in st]).mean(axis=0) for st in at]
        logger.info('Acquired sentence embedding of {} sentences.'.format(len(sentenceEmb)))
        return sentenceEmb

    @staticmethod
    def getArticleEmb(at, emb):
        """
        Average emb by article.
        - Return: array(300,)
        """
        articleEmb = np.array([emb[tk] for st in at for tk in st]).mean(axis=0)
        logger.info('Acquired article embedding of {} articles.'.format(len(articleEmb)))
        return articleEmb
