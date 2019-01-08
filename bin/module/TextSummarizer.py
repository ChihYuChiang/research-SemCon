'''
Train a recurrent convolutional network on the IMDB sentiment
classification task.
Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.

Note:
batchSize is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''
import bin.module.TextPreprocessor as TextPreprocessor
import bin.module.util as util
from bin.setting import path, textSummarizer as config

from keras.preprocessing import sequence
from keras.utils import Sequence
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, MaxPooling1D

# emb, _ = TextPreprocessor.EmbOperator.loadPretrainedEmb8Keywords(path.gNewsW2V)
dfDispatcher = TextPreprocessor.DfDispatcher(path.textIMDBDf)

class IMDBReader():

    @staticmethod
    def readAsDf():
        import pandas as pd
        import os
        import re

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
                    with open(filepath, 'rt') as f:
                        text = f.read()
                    dic['{}'.format(re.search('(.+)/(.+)\.', filepath).group(2))] = [title, rating, text, positive, group_orig]
                
                #Get title id from the url files
                elif re.match('urls', filename):
                    with open(filepath, 'rt') as f:
                        positive_url = str(not re.search('neg', filename))
                        urlFiles['{}-{}'.format(group_orig, positive_url)] = list(f) #Turn into a list, each line an element
                        
        df = pd.DataFrame.from_dict(dic, orient='index')
        df.columns = ['title', 'rating', 'text', 'positive', 'group_orig']

        print('Read in {} reviews.'.format(df.shape[0]))
        return df
    
    @classmethod
    def exportDf(cls):
        cls.readAsDf().to_csv(path.textIMDBDf)

# IMDBReader.exportDf()
# df = pd.read_csv(path.textIMDBDf)

tokenizer = TextPreprocessor.Tokenizer(dfDispatcher.getCol('text'))
articles_tokenized = tokenizer.tokenize()
with open(path.textFolder + 'IMDB_tokenized.pkl', 'wb') as f:
    pickle.dump(articles_tokenized, f)

normalizer = TextPreprocessor.Normalizer(articles_tokenized)
articles_normalized = normalizer.lower().filterStop().filterNonWord().getNormalized()


#--Data
test = SetDivider([0.1, 0.85, 0.05])
test.divideSets()
test.idSet(9)

id_train, id_test = util.divideSets([0.8, 0.2], nSample)

data = util.DataContainer({
    'train': {
        'x': (df[id] for id in id_train),
        'y': (df[id] for id in id_train)
    },
    'test': {
        'x': (df[id] for id in id_test),
        'y': (df[id] for id in id_test)
    }
})


class DataDispatcher(Sequence):

    def __init__(self, idPool, batchSize, genData):
        self.idPool = idPool
        self.batchSize = batchSize
        self.genData = genData

    def __len__(self):
        return int(np.ceil(len(self.idPool) / float(self.batchSize)))

    def __getitem__(self, idx):
        targetIds = self.idPool[idx * self.batchSize:(idx + 1) * self.batchSize]
        return self.genData(targetIds)

def genData(): pass
len(DataDispatcher([1,2,3], [1,2,3], 2, 'test'))


class Model_Sentiment():

    def __init__(self, data, **params):
        self.params = config.sentimentModelParams #Default
        self.params.update(**params)

    def preprocess(self):
        #Pad sequence
        data.train.x = sequence.pad_sequences(data.train.x, **self.params.config_padSequence)
        data.test.x = sequence.pad_sequences(data.test.x, **self.params.config_padSequence)
        print('x_train shape:', data.train.x.shape)
        print('x_test shape:', data.test.x.shape)

    def compile():
        EmbWPresetWeight = Embedding(input_dim=self.params.vocabSize, output_dim=300)
        if self.params.wmbWeightInit:
            EmbWPresetWeight.set_weights(self.params.wmbWeightInit)

        inputs = Input(shape=(self.params.config_padSequence['maxlen'], ), dtype='int32')
        _ = EmbWPresetWeight(inputs)
        _ = Dropout(self.params.dropoutRate)(_)
        _ = Conv1D(**self.params.config_conv1D, strides=1, padding='valid', activation='relu')(_)
        _ = MaxPooling1D(self.params.poolSize)(_)
        _ = LSTM(**self.params.config_LSTM)(_)
        outputs = Dense(1, activation='linear')(_)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='logcosh', optimizer='adam', metrics=['logcosh']) #TODO: customize the metric

    def train():
        model.fit(data.train.x, data.train.y, batchSize=self.params.batchSize, epochs=self.params.epochs)

    def evaluate():
        score, acc = model.evaluate(data.test.x, data.test.y, batchSize=self.params.batchSize)
        print('Test score:', score)
        print('Test accuracy:', acc)
    
    def predict():
        pass
