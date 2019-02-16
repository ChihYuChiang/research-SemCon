import os
import re
import operator
import pickle
import pandas as pd
import numpy as np
from functools import reduce
from zlib import adler32
from gensim.models import KeyedVectors
from gensim.corpora import Dictionary, HashDictionary
from nltk.probability import FreqDist
from nltk import word_tokenize, sent_tokenize

import bin.const as CONST
import bin.module.util as util
from bin.setting import path, textPreprocessor as config

logger = util.general.initLogger(loggerName='TextPreprocessor')




#TODO: pad sequence
#--Tokenize a list (generator) of articles with sentence structure
class Tokenizer():

    def __init__(self, articles):
        self.articles = articles

    def _checkExhausted(self):
        self.articles = util.general.isExhausted(iter(self.articles))
        assert self.articles , 'Articles are exhausted. Please re-initialize.'
    
    def tokenizeGen(self): #Generator only
        self._checkExhausted()
        tokenGen = ((word_tokenize(st) for st in sent_tokenize(at)) for at in self.articles)
        logger.info('Created token generator.')
        return tokenGen
    
    def tokenize(self): #Really produce the tokens
        tokenized = util.general.createListFromGen(self.tokenizeGen())
        logger.info('Tokenized articles.')
        return tokenized
    
    def tokenizeGen_st(self):
        self._checkExhausted()
        tokenGen = ((st for st in sent_tokenize(at)) for at in self.articles)
        logger.info('Created token (sentences) generator.')
        return tokenGen        

    def tokenize_st(self):
        tokenized = util.general.createListFromGen(self.tokenizeGen_st())
        logger.info('Tokenized articles into sentences.')
        return tokenized
    
    @classmethod
    def test_tokenize(cls):
        """
        Print out sample articles and tokens with sentence structure
        """
        #2 articles with 2 sentences each + an empty article
        sampleAt = ['This is a test for text preprocessing. Do you think this could be a good way to expand your knowledge?', 'Is that because theres always an inherent overhead to using classes in Python? And if so, where does the overhead come from technically speaking.', ' ']
        print(sampleAt)
        print(cls(sampleAt).tokenize())




#--Normalize a list of tokens, w/ or w/o sentence structure
class Normalizer():
    """
    Input: List of articles, with list of sentences -> [[[words], [words]]].
    - It works faster than queued operation (in a list) implementation.
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
        logger.info('Included "lower" operation in Normalizer.')
        return self

    def filterNonWord(self):
        self.gen = (((tk for tk in st if tk.isalpha()) for st in at) for at in self.gen)
        logger.info('Included "filterNonWord" operation in Normalizer.')
        return self
    
    def filterStop(self, stopwords=config.stopwords):
        self.gen = (((tk for tk in st if tk not in stopwords) for st in at) for at in self.gen)
        logger.info('Included "filterStop" operation in Normalizer.')
        return self

    def keep(self, words2Keep=config.words2Keep):
        self.gen = (((tk for tk in st if tk in words2Keep) for st in at) for at in self.gen)
        logger.info('Included "keep" operation in Normalizer.')
        return self

    def filter(self, words2Filter=config.words2Filter):
        self.gen = (((tk for tk in st if tk not in words2Filter) for st in at) for at in self.gen)
        logger.info('Included "filter" operation in Normalizer.')
        return self

    def lemmatize(self, lemmatizer=config.lemmatizer):
        self.gen = (((lemmatizer(tk) for tk in st) for st in at) for at in self.gen)
        logger.info('Included "lemmatize" operation in Normalizer.')
        return self

    def stem(self, stemmer=config.stemmer):
        self.gen = (((stemmer(tk) for tk in st) for st in at) for at in self.gen)
        logger.info('Included "stem" operation in Normalizer.')
        return self
    
    def preservePN(self, POSTagger=config.POSTagger):
        """
        Mark and combine proper nouns into "_pN_New_York" format.
        """

        ats_pos = ((POSTagger(st) for st in at) for at in self.gen)

        def _combinePN(acc, cur):
            idx, tag = cur
            tks, tmpPN, output = acc
            curTk = tks[idx]
            if tag in ['NNP', 'NNPS']:
                if not tmpPN: tmpPN.append('_pN') #Initialize when empty
                tmpPN.append(curTk)
                return (tks, tmpPN, output)
            else:
                if tmpPN:
                    output.append('_'.join(tmpPN))
                    del tmpPN[:] #Empty list instead creating new [], memory efficient
                output.append(curTk)
                return (tks, tmpPN, output)

        def _reduce(st_pos):
            tks, pos = list(zip(*st_pos))
            acc = reduce(_combinePN, enumerate(pos), (tks, [], []))
            return acc[2]

        self.gen = ((_reduce(st_pos) for st_pos in at_pos) for at_pos in ats_pos)
        logger.info('Included "preservePN" operation in Normalizer.')
        return self

    @classmethod
    def test_operations(cls):
        """
        Test chained preservePN(), lower(), filterStop(), stem()
        """
        tokens = [[['test', 'Is', 'gooD', '.'], ['HELLO', 'Kitty', 'world']], [['overhead', 'comes', 'from', 'technically', 'speaking']], []]
        print(tokens)
        print(cls(tokens).preservePN().lower().filterStop().stem().getNormalized())




class Mapping():

    _idRange = 2000000
    test_tokens = [[['test', 'is', 'good', '.'], ['good', 'test'], ['hello', 'test']], [['overhead', 'comes', 'good', 'is', 'speaking']], []]
    
    def __init__(self, tokens=[], idRange=_idRange):
        self.tokens = tokens
        self.idRange = idRange
        self.dict = Dictionary()

    def makeDict(self, bidirectional=True):
        self.dict = Dictionary((util.general.flattenList(at) for at in self.tokens), prune_at=self.idRange)
        if bidirectional:
            #The `id2token` property is lazily computed. Use the `get` to force producing the revered dict
            self.dict.get(0)
        logger.info('Created token dictionary.')
    
    def brief(self):
        #Word frequency distribution by nltk
        fdist = FreqDist((tk for at in self.tokens for st in at for tk in st))

        nArticle = len(self.tokens)
        nSentence = sum(map(lambda article: len(article), self.tokens))
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
    
    def getBOW(self, article):
        '''
        Transform an article into BOW format.
        '''
        return self.dict.doc2bow(util.general.flattenList(article))
    
    @staticmethod
    def hash(token, idRange=_idRange):
        '''
        For hashing trick
        - Use `hash` or `md5` if the dictionary is very large.
        - Note that `hash` is unstable across different runs.
        - Sample code:
            # hash('test')
            # from hashlib import md5
            # int(md5(b'test').hexdigest(), 16)
        '''
        return adler32(bytes(token, 'utf-8')) % idRange
    
    @classmethod
    def test_makeDict(cls):
        mapping = cls(cls.test_tokens)
        mapping.makeDict()
        print(cls.test_tokens)
        print(mapping.dict.id2token)
        print(mapping.dict.token2id)

    @classmethod
    def test_brief(cls):
        print(cls.test_tokens)
        cls(cls.test_tokens).brief()




#--Load experience keywords and embedding operations
class EmbOperation():

    @staticmethod
    def loadKeywords(path):
        """
        Load experience keywords.
        """
        keyWords = pd.read_csv(path.expKeyword, encoding='utf-8', header=None)[0].tolist()
        keyWords = [word.lower() for word in keyWords]

        logger.info('Loaded exp keywords of', len(keyWords), 'words.')
        return keywords
    
    @staticmethod
    def loadPretrainedEmb(path):
        """
        Load pretrained emb
        - Note for GoogleNews: w/o punctuation, w/o stemming, w stopwords, w some n-grams
        """
        logger.info('Loading pretrained embedding at "{}".'.format(path))
        emb = KeyedVectors.load_word2vec_format(path, binary=True)

        #Provide the KeyError fallback `get` function
        def getEmb(key):
            try: return emb[key]
            except KeyError: return None
        emb.get = getEmb

        logger.info('Loaded pretrained embedding with {} terms.'.format(str(len(emb.index2word))))
        return emb
    
    @staticmethod
    def sliceEmb(vocabulary, emb):
        """
        - Take only part of a big emb, leaving out terms not in the target corpus.
        - If the term is not included in the emb, use zero vector instead.
        - Return: np 2d array with the same order of vocabulary.
        """
        embMatrix = []
        for term in vocabulary:
            try:
                embMatrix.append(emb[term])
            except KeyError:
                embMatrix.append(np.zeros(emb.vector_size))
                continue
        embMatrix = np.array(embMatrix)

        logger.info('Acquired embedding matrix of {} terms.'.format(str(len(vocabulary))))
        return embMatrix
        
    @staticmethod
    def getSentenceEmb(at, emb):
        """
        Average emb by sentence.
        - Return: [sent=array(300,)]
        """
        sentenceEmb = [np.array([emb.get(tk) for tk in st if emb.get(tk) is not None]).mean(axis=0) for st in at]

        logger.info('Acquired sentence embedding of {} sentences.'.format(str(len(sentenceEmb))))
        return sentenceEmb

    @staticmethod
    def getArticleEmb(at, emb):
        """
        Average emb by article.
        - Return: array(300,)
        """
        articleEmb = np.array([emb.get(tk) for st in at for tk in st if emb.get(tk) is not None]).mean(axis=0)

        logger.info('Acquired article embedding of {} articles.'.format(str(len(articleEmb))))
        return articleEmb




class IMDBReader():

    @staticmethod
    def readAsDf():
        dic = {}
        urlFiles = {}
        for (dirpath, dirnames, filenames) in os.walk(path.textIMDBFolder, topdown=True):
            group_orig = 'train' if re.search('train', dirpath) else 'test' #The original train/test seperation by the author
            positive = not re.search('neg', dirpath)
            urlFile = '{}-{}'.format(group_orig, str(positive))
            
            for filename in filenames:
                match = re.match('^(\d+)_(\d+)\.txt$', filename)
                filepath = os.path.join(dirpath, filename)
                
                #If the particular filename format, read txt file into df
                if match:
                    title = re.search('title/(.+)/', urlFiles[urlFile][int(match.group(1))]).group(1)
                    rating = match.group(2)
                    with open(filepath, 'rt', encoding='utf-8') as f:
                        text = f.read()
                    dic['{}'.format(re.search('(.+)/(.+)\.', filepath).group(2))] = [title, rating, text, positive, group_orig]
                
                #Get title id from the url files
                elif re.match('urls', filename):
                    with open(filepath, 'rt') as f:
                        positive_url = str(not re.search('neg', filename))
                        urlFiles['{}-{}'.format(group_orig, positive_url)] = list(f) #Turn into a list, each line an element
                        
        df = pd.DataFrame.from_dict(dic, orient='index')
        df.columns = ['title', 'rating', 'text', 'positive', 'group_orig']

        logger.info('Read in {} reviews.'.format(df.shape[0]))
        return df
    
    @classmethod
    def exportDf(cls):
        cls.readAsDf().to_csv(path.textIMDBDf, encoding='utf-8')
        logger.info('Export the df at {}.'.format(path.textIMDBDf))




def saveTokens(tokenized, normalized, desc):
    with open('{}tokenized_{}.pkl'.format(path.textTkFolder, desc), 'wb') as f: pickle.dump(tokenized, f)
    with open('{}normalized_{}.pkl'.format(path.textTkFolder, desc), 'wb') as f: pickle.dump(normalized, f)
    logger.info('Saved tokenized and normalized {} text at {}.'.format(desc, path.textTkFolder))