class SetDivider():
    """
    Acquire ids of arbitrary set division
    - Given proportion and the number of samples, return ids (starts from 0) of each set [set, set, ...].
    - An element in `proportion` represents a set.
    - The proportion must sum to 1.
    - If sample size is not provided, return base sample size and ids represent the proportion. Use % to find the remainders for assigning sets.
    - Use `idSet` to identify the set identity of a sample index.
    """
    import numpy as np
    import re

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
        ids = []
        for nId in nIds:
            id = []
            while nId > 0:
                try: id.append(next(rIndiceGen))
                except StopIteration: break
                nId -= 1
            ids.append(id)

        return ids

    def _nSampleWO(self):
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
    """
    import pandas as pd

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

        print('Initiated df dispatcher of \"{}\" from row {} to row {}.'.format(filePath, startRow, endRow or 'file ends'))

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


from keras.utils import Sequence
class KerasDataDispatcher(Sequence):
    '''
    Prepare data for Keras models with data generated on the go
    - If `shuffle == True`, `idPool` will be shuffled for each epoch. Meaning, the order of the samples go through the network will be different.
    - If `genData` uses generators, `idPool` has to be sequential--`shuffle` has to be `False`.
    '''
    import numpy as np

    def __init__(self, sampleSize, batchSize, genData, shuffle=False):
        self.idPool = list(range(sampleSize))
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
        if self.shuffle: np.random.shuffle(self.idPool)