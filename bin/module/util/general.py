#--Decorators for classes
class ClsDecorator():

    @staticmethod
    def prohibitAttrSetter(cls):
        """
        Prohibit access to attribute setter
        """
        def setattr(self, key, value):
            class ProhibittedOperation(Exception): pass
            raise ProhibittedOperation('Not allowed to modify attributes directly.')

        cls.__setattr__ = setattr
        return cls
    
    @staticmethod
    def grantKeywordUpdate(cls):
        """
        Grant attribute modification by `update` method
        """
        def update(self, **kwarg):
            for key, value in kwarg.items():
                self.__dict__[key] = value

        cls.update = update
        cls.__init__ = update     
        return cls


#--Decorators for functions
class FuncDecorator():
    
    @staticmethod
    def delayOperation(time):
        """
        Delay operation by `time` secs
        - 0.7*time + 0.6*random()*time
        - When time=10, it's 7-13 secs
        """
        from time import sleep
        from random import random
        def wrapper(original_function):
            def new_function(*args, **kwargs):
                sleep(0.7 * time + 0.6 * random() * time)

                original_output = original_function(*args, **kwargs)   
                return original_output
            return new_function
        return wrapper
    

class UniversalContainer():
    """
    Usage
    - Print object to see all key and data (recursive). Maximum print len for each item is 100.
    - getKeys() shows all attribute keys of this object (current level).
    - getMethods() shows all methods of this object.
    - get() works as the same method of a dict.
    """

    def __repr__(self, level=0):
        import pandas as pd

        keys = [item for item in dir(self) if not callable(getattr(self, item)) and not item.startswith("__")]
        rep = []

        for key in keys:
            attr = getattr(self, key)
            if isinstance(attr, UniversalContainer):
                rep.append('-' * 3 * level + '.' + key)
                rep.append(attr.__repr__(level + 1))
            else:
                #Print index only if a pandas df
                attrStr = str(attr.keys()) if isinstance(attr, pd.core.frame.DataFrame) else str(attr)
                rep.append('-' * 3 * level + '.' + key)
                rep.append('-' * 3 * level + ' ' + (attrStr[:100] + ' \u2026' if len(attrStr) > 100 else attrStr)) #Unicode for horizontal ellipsis

        return '\n'.join(rep)

    def getKeys(self):
        return [item for item in dir(self) if not callable(getattr(self, item)) and not item.startswith("__")]

    def getMethods(self):
        return [item for item in dir(self) if callable(getattr(self, item)) and not item.startswith("__")]
    
    def get(self, key, default=None):
        return self[key] if key in dir(self) else default


class DataContainer(UniversalContainer):
    """
    Usage
    - Pass a dict when initiate to create a nested DataContainer structure with default values. It stops at none-dict objs.
    - Eg. var = DataContainer({'a': {'a_1': {}, 'a_2': 'foo'}, 'b': [1, 2]}).
    - If want to keep dict structures, create and empty container and assign later.
    """
    
    def __new__(cls, structure={}):
        if isinstance(structure, dict):
            return super().__new__(cls)
        else: return structure
        
    def __init__(self, structure={}):
        for key, item in structure.items():
            self.__dict__[key] = type(self)(item)


@ClsDecorator.prohibitAttrSetter
@ClsDecorator.grantKeywordUpdate
class SettingContainer(UniversalContainer):
    """
    Usage
    - Convenient keyword = parameter setting.
    - Protected attribute setter. Use `update(key=value)` to modify content.
    - Compatible with `**self` expression
    """

    #The keys and __getitem__ make the obj pass as a mapping obj -> can be used with ** expression
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        return self.__dict__[key]
   

class ConvertedContainer(SettingContainer):
    """
    Usage
    - Convert dict to object form (recursive).
    """

    def __new__(cls, data):
        from collections import Iterable

        if isinstance(data, dict):
            return super().__new__(cls) #Pass 'class instance' upward for inheritance
        elif isinstance(data, Iterable) and not isinstance(data, str):
            return type(data)(cls(value) for value in data)
        else:
            return data

    def __init__(self, data):
        for i in range(len(data.keys())):
            self.__dict__[list(data.keys())[i]] = type(self)(list(data.values())[i])


class Session(UniversalContainer):
    """
    Conveniently storage of session info
    - Use 'load' class method to load session file with specified path.
    - Use key=value pairs to give initial values to session attributes.
    - If the session file is successfully loaded, the key=value pairs will be ignored.
    - Use 'dump' instance method to save session file with specified path.
    """

    def dump(self, path):
        import pickle

        with open(path, 'wb') as f: pickle.dump(self, f)
        print('Dumped session at \'{}\'.'.format(path))
    
    @classmethod
    def load(cls, path, **kwarg):
        import pickle

        try:
            with open(path, 'rb') as f:
                session = pickle.load(f, encoding='utf-8')
                print('Loaded session at \'{}\'.\n{}'.format(path, session))
        except:
            session = cls()
            for key, value in kwarg.items():
                session.__dict__[key] = value
            print('Did not find session at \'{}\'. Initiated a new session.\n{}'.format(path, session))

        return session




def getConfigObj(path):
    """
    Read config files into an obj container
    - Support file type: .json, .yml.
    """
    import yaml
    import re

    class UnknownFileType(Exception): pass

    with open(path, 'r', encoding='utf-8') as f:
        ext = re.search('\.(.+)', path).group(1)
        if ext == 'json': config_dic = json.load(f)
        elif ext == 'yml': config_dic = yaml.load(f)
        else: raise UnknownFileType('\'{}\' is not a supported file type.'.format(ext))

    return ConvertedContainer(config_dic)


def writeJsls(obj, path):
    """
    Write all objs of a iterable into a jsl file
    """
    import json
    import numpy

    #Deal with the json default encoder defect
    #https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, numpy.integer):
                return int(obj)
            elif isinstance(obj, numpy.floating):
                return float(obj)
            elif isinstance(obj, numpy.ndarray):
                return obj.tolist()
            else:
                return super(NumpyEncoder, self).default(obj)
            
    with open(path, mode='a') as f:
        for item in obj:
            json.dump(item, f, cls=NumpyEncoder)
            f.write('\n')
    
    print('Completed writing \'{}\', appended obj len {}.'.format(path, len(obj)))


def readJsls(path):
    """
    Read all objs in one jsl file
    """
    import json

    output = []
    with open(path, mode='r') as f:
        for line in f:
            output.append(json.loads(line))
    
    print('Completed reading \'{}\', loaded obj len {}.'.format(path, len(output)))
    return output


def initLogger(loggerName, file='INFO', console='DEBUG', slack=None):
    """
    Initialize a logger using logging module
    - INFO or above will be saved in file.
    - DEBUG or above will be printed in console.
    - Slack handler can be configured.
    - slack = {
        'level': 'WARNING',
        'host': 'slack.com',
        'url': '/api/chat.postMessage',
        'token': cred.Slack.token,
        'channel': '#general'
    }
    - https://docs.python.org/3/library/logging.html#logging-levels.
    """
    import logging

    #Create new logger
    logger = logging.getLogger(loggerName)
    logger.setLevel(logging.DEBUG)
    logger.handlers = [] #Remove existing handlers

    #Formatter reference
    #'%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    #Create file handler and add to logger
    if file:
        fh = logging.FileHandler('log/{}.log'.format(loggerName), mode='w+')
        fh.setLevel(file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    
    #Create console handler and add to logger
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(console)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
    
    #Create http handler for Slack integration
    #https://api.slack.com/methods/chat.postMessage
    if slack:
        cred = getConfigObj('ref/credential.yml')
        sh = logging.handlers.HTTPHandler(host=slack.pop('host'), url=slack.pop('url'), method='POST', secure=True)
        sh.setLevel(slack.pop('level'))
        def mapLogRecord(record):
            return {
                **slack,
                'text': record.__dict__['message'],
                'username': 'Python Logger - ' + loggerName
            }
        sh.mapLogRecord = mapLogRecord
        logger.addHandler(sh)

    return logger


def createListFromGen(generator):
    """
    Transform a generator into a list (recursive)
    """
    from types import GeneratorType

    if isinstance(generator, GeneratorType):
        return [createListFromGen(i) for i in generator]
    else:
        return generator


def flattenList(l, nLevel=-1):
    """
    Flatten list (recursive for `nLevel`)
    - Parameter: `l`, a list; `nLevel`=-1 if extract all levels
    - Return: a flattened list as a generator
    """
    import collections

    for el in l:
        if isinstance(el, collections.Sequence) and not isinstance(el, (str, bytes)) and nLevel != 0:
            yield from flattenList(el, nLevel - 1)
        else:
            yield el


def createCustomHeader():
    """
    Create customized HTTP header with random user agent
    - The `agent` candidates have to be updated manually by the information provided in the link.
    - https://techblog.willshouse.com/2012/01/03/most-common-user-agents/.
    - Default `accept` types for each browser can be found in the link.
    - https://developer.mozilla.org/en-US/docs/Web/HTTP/Content_negotiation/List_of_default_Accept_values
    """
    import random

    uaCandidates = [#Acquire updated user agents from: 
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134'
    ]
    
    #Randomly select one from the candidates and update request header
    headers = {
        'User-Agent': random.choice(uaCandidates),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive',
        'Referer': 'https://www.google.com/',
        'Upgrade-Insecure-Requests': '1',
        'X-Https': '1'
    }

    return headers


def makeDirAvailable(directory):
    """
    Make sure the directory is available. If not, create one. (including all intermediate-level directories)
    """
    import os

    if not os.path.exists(directory):
        os.makedirs(directory)


def isExhausted(iterable, marker=True):
    """
    Examine if the iterable (eg. generator) is empty
    - if empty (exhausted), return `marker` or raise `StopIteration`.
    - if not empty, returns the iterable.
    """
    import itertools

    try:
        firstItem = next(iterable)
    except StopIteration:
        if marker: return marker
        else: raise

    return itertools.chain([firstItem], iterable)