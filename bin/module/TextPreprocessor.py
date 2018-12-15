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


#--Tokenize a list (generator) of articles with sentence structure
class Tokenizer():

    def __init__(self, articles=''):
        self.result = articles
    
    def tokenize(self):
        self.result = ((nltk.word_tokenize(st) for st in nltk.sent_tokenize(at)) for at in self.articles)
    
    def brief(self, tokens=None):
        """
        Input: tokens with sentence structure. Default to `self.result`.
        """
        if tokens is None: tokens = self.result

        #Word frequency distribution by nltk
        fdist = FreqDist((tk for at in tokens for st in at for tk in st))

        nArticle = len(tokens)
        nSentence = sum(map(lambda article: len(article), tokens))
        nWord = fdist.N()

        print('About the corpus')
        print('- Number of articles:', nArticle)
        print('- Number of sentences:', nSentence)
        print('- Number of terms:', nWord)
        print('- Number of unique terms:', fdist.B())
        print('- Top terms:', sorted(fdist.items(), key=operator.itemgetter(1), reverse=True)[0:5])
        print('- Terms per sentence:', nWord / nSentence)
        print('- Terms per article:', nWord / nArticle)
        print('- Sentences per article:', nSentence / nArticle)
    
    @classmethod
    def test_tokenize(cls):
        """
        Print out sample articles and tokens with sentence structure
        """
        #2 articles with 2 sentences each
        articles = ['This is a test for text preprocessing. Do you think this could be a good way to expand your knowledge?', 'Is that because theres always an inherent overhead to using classes in Python? And if so, where does the overhead come from technically speaking.']
        print(articles)
        print(util.createListFromGen(cls(articles).tokenize().result[1]))
    
    @classmethod
    def test_brief(cls):
        tokens = [[['test', 'Is', 'gooD', '.'], ['gooD', 'world']], [['overhead', 'comes', 'from', 'gooD', 'speaking']]]
        print(tokens)
        cls().brief(tokens)


#--Normalize a list of tokens, w/ or w/o sentence structure
class Normalizer():
    """
    Input: List of articles, with list of sentences -> [[[words], [words]]].
    """

    def __init__(self, tokens):
        self.result = tokens
    
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
        print(util.createListFromGen(cls(tokens).lower().filterStop().stem().result))


#--A df row generator
class DfDispatcher():
    """
    Read in by chunk (save disk access times) but yield by row
    - `chunkSize` = how many rows to read per access.
    - Dispatch between `startRow` and `endRow` (inclusive).
    - Return (rowId, rowContent) for each row.
    """

    def __init__(self, filePath, startRow=0, endRow=9, chunkSize=1000):
        #"cp1252", "ISO-8859-1", "utf-8"
        self.readCsvParam = {
            'filepath_or_buffer': filePath,
            'encoding': 'cp1252',
            'chunksize': chunkSize,
            'nrows': 1 + endRow
        }
        self.startRow = startRow
        self.dfIter = self.__iter__()

    def __iter__(self):
        dfIter = (row for chunk in pd.read_csv(**self.readCsvParam) for row in chunk.iterrows())
        i = 0
        while i < self.startRow:
            i += 1
            next(dfIter)
        return dfIter

    def __next__(self):
        return next(self.dfIter)
    
    def getCol(self, colName):
        for i, row in self:
            yield row[colName]


#--Load experience keywords and embedding operations
class EmbOperator():
    
    def loadPretrainedEmb8Keywords(path):
        """
        Load pretrained emb at `path` and exp keywords
        """
        emb = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        print('Loaded pretrained embedding with', len(emb.index2word), 'words.')

        keyWords = pd.read_csv(path.expKeyword, encoding='utf-8', header=None)[0].tolist()
        keyWords = [word.lower() for word in keyWords]
        print('Load exp keywords of', len(keyWords), 'words.')

        return (emb, keywords)

    def getSentenceEmb(at, emb):
        """
        Average emb by sentence.
        - Return: [sent=array(300,)]
        """
        return [np.array([emb[tk] for tk in st]).mean(axis=0) for st in at]

    def getArticleEmb(at, emb):
        """
        Average emb by article.
        - Return: array(300,)
        """
        return np.array([emb[tk] for st in at for tk in st]).mean(axis=0)
