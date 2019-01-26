import numpy as np


class SetDivider():
    """
    Acquire ids of arbitrary set division
    - Given proportion and the number of samples, return ids (starts from 0) of each set [set, set, ...].
    - An element in `proportion` represents a set.
    - The proportion must sum to 1.
    - If sample size is not provided, return base sample size and ids represent the proportion. Use % to find the remainders for assigning sets.
    - Use `idSet` to identify the set identity of a sample index.
    """

    def __init__(self, proportion, nSample=None, seed=1):
        assert sum(proportion) == 1, '`proportion` must sum to 1.'
        self.proportion = proportion
        self.nSample = nSample
        self.ids = []

        #Reset np seed
        np.random.seed(seed=seed)

    def _nSampleW(self):
        #Number of indice for each set
        nIds = np.around(np.array(self.proportion) * self.nSample)

        #Shuffle the indice pool
        rIndiceGen = np.arange(self.nSample)
        np.random.shuffle(rIndiceGen)
        rIndiceGen = (i for i in rIndiceGen)

        #Assign indice to each set
        ids = tuple()
        for nId in nIds:
            id = []
            while nId > 0:
                try: id.append(next(rIndiceGen))
                except StopIteration: break
                nId -= 1
            ids += (id,)

        return ids

    def _nSampleWO(self):
        import re
        
        #Max number of float digits
        digits = max(map(lambda x: len(re.search('\.(\d+)', str(x)).group(1)), self.proportion))

        #Indice pool and randomized base ids
        self.nSample = 10 ** digits
        ids_base = self._nSampleW()

        return ids_base

    def divideSets(self):
        if self.nSample is not None: self.ids = self._nSampleW()
        else: self.ids = self._nSampleWO()
        return self.ids, self.nSample

    def idSet(self, sampleIdx):
        targetIdx = sampleIdx % self.nSample
        for i in range(len(self.proportion)):
            if targetIdx in self.ids[i]: return i
                

class DfDispatcher():
    """
    Read in by chunk (save disk access times) but yield by row
    - `chunkSize` = how many rows to read per access.
    - Dispatch between `startRow` and `endRow` (inclusive).
    - Return (rowId, rowContent) for each row.
    - Useful encodings: "cp1252", "ISO-8859-1", "utf-8".
    - `targetCols = where if NA exists, that row will be dropped.
    """

    def __init__(self, filePath, startRow=0, endRow=None, encoding="utf-8", chunkSize=1000, targetCols=None):
        self.targetCols = targetCols
        self.readCsvParam = {
            'filepath_or_buffer': filePath,
            'encoding': encoding,
            'chunksize': chunkSize,
            'nrows': 1 + endRow if endRow else None
        }
        self.startRow = startRow
        self.dfIter = self.__iter__()

        print('Initiated df dispatcher of \"{}\" from row {} to row {}.'.format(filePath, startRow, endRow or 'file ends'))

    def __iter__(self):
        import pandas as pd

        dfIter = (row for chunk in pd.read_csv(**self.readCsvParam) for row in chunk.dropna(subset=self.targetCols).iterrows())
        i = 0
        #TODO: try use send() instead
        while i < self.startRow:
            i += 1
            next(dfIter)
        return dfIter

    def __next__(self):
        return next(self.dfIter)
    
    def getCol(self, colName):
        col = (row[colName] for i, row in self)
        self.dfIter = self.__iter__() #Reinitialize the generator

        return col 


from keras.utils import Sequence
class KerasDataDispatcher(Sequence):
    '''
    Prepare data for Keras models with data generated on the go
    - `genData` accepts a list of targetIds and return real data.
    - If `shuffle == True`, `idPool` will be shuffled for each epoch. Meaning, the order of the samples go through the network will be different.
    - If `genData` data sources are generators, the model will not be able to multiprocessing.
    '''

    def __init__(self, sampleSize, batchSize, genData, shuffle=False):
        self.idPool = np.arange(sampleSize)
        self.batchSize = batchSize
        self.genData = genData
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.idPool) / float(self.batchSize)))

    def __getitem__(self, idx):
        targetIds = self.idPool[idx * self.batchSize:(idx + 1) * self.batchSize]
        return self.genData(targetIds)
    
    def on_epoch_end(self):
        """
        Shuffle index after each epoch.
        """
        if self.shuffle == True: np.random.shuffle(self.idPool)


from abc import ABC, abstractmethod
class KerasModelBase(ABC):
    """
    Abstract class for implementing Keras model.
    - Handle `model`, `params`, and common operations.
    - When inherit, must also inherit `KerasModel` or `KerasModelGen`.
    """

    def __init__(self, params):
        import bin.module.util as util
        
        self.model = object()
        self.params = util.general.SettingContainer(
            batchSize = 32,
            config_multiprocessing = {},
            config_compile = {}, config_training = {},
            config_evaluate = {}, config_predict = {}
        )
        self.params.update(**params)
    
    @abstractmethod
    def compile(self, inputs, outputs):
        from keras.models import Model

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(**self.params.config_compile)
        self.model.summary()

    def save(self, path, **other):
        """
        `path` includes the folder containing 3 files pertaining to the model.
        """
        import pickle
        util.general.createFolder(path)

        #Config (only the graph)
        config = model.to_yaml()
        with open(path + 'config.yaml', "w") as f:
            f.write(config)

        #Weights
        model.save_weights(path + 'weights.h5')

        #Other stuffs
        with open(path + 'supplements.pkl', 'wb') as f:
            pickle.dump((self.params, {**other}), f)
    
    def load(self, path):
        import pickle
        from keras.models import Model

        #Config (only the graph)
        with open(path + 'config.yaml', 'r') as f:
            config = f.read()

        #Other stuffs
        with open(path + 'supplements.pkl', 'rb') as f:
            self.params, other = pickle.load(f)
        for key, value in other.items():
            self.__dict__[key] = value

        self.model = model_from_yaml(config)
        self.model.load_weights(path + 'weights.h5')
        self.model.compile(**self.params.config_compile)
        self.model.summary()


class KerasModel(ABC):
    """
    Real data (numpy arrays) as input.
    """

    @abstractmethod
    def train(self, x_train, y_train):
        trainingHistory = self.model.fit(x_train, y_train, batch_size=self.params.batchSize, **self.params.config_training)
        return trainingHistory

    @abstractmethod
    def evaluate(self, x_test, y_test):
        metrics = self.model.evaluate(x_test, y_test, batch_size=self.params.batchSize, **self.params.config_evaluate)
        return metrics

    @abstractmethod    
    def predict(self, x_new):
        prediction = self.model.predict(x_new,
            batch_size=self.params.batchSize, **self.params.config_predict)
        return prediction


class KerasModelGen(ABC):
    """
    Generator as input, each iterate as a batch.
    """

    @abstractmethod
    def train(self, gen_train):
        trainingHistory = self.model.fit_generator(
            generator=gen_train,
            **self.params.config_multiprocessing,
            **self.params.config_training
        )
        return trainingHistory

    @abstractmethod
    def evaluate(self, gen_test):
        metrics = self.model.evaluate_generator(
            generator=gen_test,
            **self.params.config_multiprocessing,
            **self.params.config_evaluate
        )
        return metrics

    @abstractmethod    
    def predict(self, gen_new):
        prediction = self.model.predict(
            generator=gen_new,
            **self.params.config_multiprocessing,
            **self.params.config_predict
        )
        return prediction




def ids2Onehot(ids, vocabSize):
    """
    - Input: a list of word index.
    - Output: a onehot numpy array with dimension (len(ids), vocabSize 
    """
    m, n = len(ids), vocabSize
    onehot = np.zeros((m, n))
    onehot[np.arange(m), np.array(ids)] = 1

    return onehot