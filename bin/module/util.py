#--Decorators for classes
class ClsDecorator():

    def prohibitAttrSetter(cls):
        """
        Prohibit access to attribute setter
        """
        def setattr(self, key, value):
            class ProhibittedOperation(Exception): pass
            raise ProhibittedOperation('Not allowed to modify attributes directly.')

        cls.__setattr__ =  setattr
        return cls
    
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
    
    def delayOperation(time):
        """
        Delay operation by `time` secs
        - 0.7*time + 0.6*random()*time
        - When time=10, it's 7-13 secs
        """
        from time import sleep
        from random import random
        def wrapper(original_function):
            def new_function(*args,**kwargs):
                sleep(0.7 * time + 0.6 * random() * time)

                original_output = original_function(*args,**kwargs)   
                return original_output
            return new_function
        return wrapper


class UniversalContainer():
    """
    Usage
    - Print object to see all key and data (recursive).
    - listKey() shows all attribute keys of this object (current level).
    - listMethod() shows all methods of this object.
    """

    def __repr__(self, level=0):
        keys = [item for item in dir(self) if not callable(getattr(self, item)) and not item.startswith("__")]
        rep = []

        for key in keys:
            attr = getattr(self, key)
            if isinstance(attr, UniversalContainer):
                rep.append('-' * 3 * level + '.' + key)
                rep.append(attr.__repr__(level + 1))
            else:
                rep.append('-' * 3 * level + '.' + key)
                rep.append('-' * 3 * level + ' ' + str(attr))

        return '\n'.join(rep)

    def listKey(self):
        return [item for item in dir(self) if not callable(getattr(self, item)) and not item.startswith("__")]

    def listMethod(self):
        return [item for item in dir(self) if callable(getattr(self, item)) and not item.startswith("__")]


@ClsDecorator.prohibitAttrSetter
@ClsDecorator.grantKeywordUpdate
class SettingContainer(UniversalContainer):
    """
    Usage
    - Convenient keyword = parameter setting.
    - Protected attribute setter. Use `update(key=value)` to modify content.
    """
    pass
   

class ConvertedContainer(SettingContainer):
    """
    Usage
    - Convert dict to object form (recursive).
    """
    
    def __new__(cls, data):
        from collections import Iterable

        if isinstance(data, dict):
            return super().__new__(cls)
        elif isinstance(data, Iterable) and not isinstance(data, str):
            return type(data)(ConvertedContainer(value) for value in data)
        else:
            return data

    def __init__(self, data):
        for i in range(len(data.keys())):
            self.__dict__[list(data.keys())[i]] = ConvertedContainer(list(data.values())[i])


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
    
    print('Completed writing {}, appended obj len {}.'.format(path, len(obj)))


def readJsls(path):
    """
    Read all objs in one jsl file
    """
    import json

    output = []
    with open(path, mode='r') as f:
        for line in f:
            output.append(json.loads(line))
    
    print('Completed reading {}, loaded obj len {}.'.format(path, len(output)))
    return output


def initLogger(loggerName, console=True, consoleLevel='DEBUG', fileLevel='INFO'):
    """
    Initialize a logger using logging module
    - INFO or up will be saved in file.
    - DEBUG or up will be printed in console.
    - https://docs.python.org/3/library/logging.html#logging-levels.
    - More information is logged in log file than in console. 
    """
    import logging

    #Create new logger
    logger = logging.getLogger(loggerName)
    logger.setLevel(logging.DEBUG)

    #Formatter reference
    #'%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    #Create file handler and add to logger
    fh = logging.FileHandler('log/{}.log'.format(loggerName), mode='w+')
    fh.setLevel(fileLevel)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    #Create console handler and add to logger
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(consoleLevel)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    return logger