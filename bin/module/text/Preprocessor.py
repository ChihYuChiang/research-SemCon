import os
import pickle
import operator
import nltk
import gensim
import pandas as pd
import numpy as np
from typing import List
from nltk.probability import FreqDist

import bin.module.util as util
from bin.setting import path, textPreprocessor as config

logger = util.general.initLogger(loggerName='TextPreprocessor')




#TODO: pad sequence
#--Tokenize a list (generator) of articles with sentence structure
class Tokenizer():

    #2 articles with 2 sentences each
    sampleAt = ['This is a test for text preprocessing. Do you think this could be a good way to expand your knowledge?', 'Is that because theres always an inherent overhead to using classes in Python? And if so, where does the overhead come from technically speaking.']

    def __init__(self, articles):
        self.articles = articles
        self.tokenized = []
    
    def tokenizeGen(self): #Generator only
        assert util.general.isExhausted(self.articles), 'Articles are exhausted. Please re-initialize.'

        tokenGen = ((nltk.word_tokenize(st) for st in nltk.sent_tokenize(at)) for at in self.articles)
        logger.info('Create token generator.')
        return tokenGen
    
    def tokenize(self): #Really produce the tokens
        if not self.tokenized:
            assert util.general.isExhausted(self.articles), 'Articles are exhausted. Please re-initialize.'

            self.tokenized = util.general.createListFromGen(self.tokenizeGen())
            logger.info('Tokenized articles.')
        return self.tokenized

    def brief(self):
        """
        Input: tokens with sentence structure.
        """
        self.tokenize()

        #Word frequency distribution by nltk
        fdist = FreqDist((tk for at in self.tokenized for st in at for tk in st))

        nArticle = len(self.tokenized)
        nSentence = sum(map(lambda article: len(article), self.tokenized))
        nWord = fdist.N()

        logger.info('About the corpus:')
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
        print(cls(cls.sampleAt).tokenize())
    
    @classmethod
    def test_brief(cls):
        cls(cls.sampleAt).brief()




#--Normalize a list of tokens, w/ or w/o sentence structure
class Normalizer():
    """
    Input: List of articles, with list of sentences -> [[[words], [words]]].
    """

    def __init__(self, tokens):
        self.gen = tokens
    
    def getGen(self):
        logger.info('Create normalization generator.')
        return self.gen

    def getNormalized(self):
        normalized = util.general.createListFromGen(self.gen)
        logger.info('Normalized articles.')
        return normalized
    
    def lower(self):
        self.gen = (((tk.lower() for tk in st) for st in at) for at in self.gen)
        return self

    def filterNonWord(self):
        self.gen = (((tk for tk in st if tk.isalpha()) for st in at) for at in self.gen)
        return self
    
    def filterStop(self, stopwords=config.stopwords):
        self.gen = (((tk for tk in st if tk not in stopwords) for st in at) for at in self.gen)
        return self

    def keep(self, words2Keep=config.words2Keep):
        self.gen = (((tk for tk in st if tk in words2Keep) for st in at) for at in self.gen)
        return self

    def filter(self, words2Filter=config.words2Filter):
        self.gen = (((tk for tk in st if tk not in words2Filter) for st in at) for at in self.gen)
        return self

    def lemmatize(self, lemmatizer=config.lemmatizer):
        self.gen = (((lemmatizer(tk) for tk in st) for st in at) for at in self.gen)
        return self

    def stem(self, stemmer=config.stemmer):
        self.gen = (((stemmer(tk) for tk in st) for st in at) for at in self.gen)
        return self
    
    @classmethod
    def test_operations(cls):
        """
        Test chained lower(), filterStop(), stem()
        """
        tokens = [[['test', 'Is', 'gooD', '.'], ['HELLO', 'world']], [['overhead', 'comes', 'from', 'technically', 'speaking']]]
        print(tokens)
        print(cls(tokens).lower().filterStop().stem().getNormalized())




#--Load experience keywords and embedding operations
class EmbOperation():
    
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
