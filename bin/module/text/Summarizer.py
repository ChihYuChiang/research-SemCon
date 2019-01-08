import pandas as pd
import os
import re

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, MaxPooling1D

import bin.module.text.Preprocessor as TextPreprocessor
import bin.module.util as util
from bin.setting import path, textSummarizer as config




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




#--Data
# IMDBReader.exportDf()
# df = pd.read_csv(path.textIMDBDf)
# emb, _ = TextPreprocessor.EmbOperator.loadPretrainedEmb8Keywords(path.gNewsW2V)

tokenizer = TextPreprocessor.Tokenizer(dfDispatcher.getCol('text'))
articles_tokenized = tokenizer.tokenize()
with open(path.textFolder + 'IMDB_tokenized.pkl', 'wb') as f:
    pickle.dump(articles_tokenized, f)

normalizer = TextPreprocessor.Normalizer(articles_tokenized)
articles_normalized = normalizer.lower().filterStop().filterNonWord().getNormalized()

id_train, id_test = util.data.SetDivider.divideSets([0.8, 0.2], nSample)
data = util.general.DataContainer({
    'train': {
        'x': (df[id] for id in id_train),
        'y': (df[id] for id in id_train)
    },
    'test': {
        'x': (df[id] for id in id_test),
        'y': (df[id] for id in id_test)
    }
})
util.general.DataContainer({
    'train': {
        'x': (df[id] for id in id_train),
        'y': (df[id] for id in id_train)
    },
    'test': {
        'x': (df[id] for id in id_test),
        'y': (df[id] for id in id_test)
    }
})


def genDataSequential(targetIds, X, Y):
    from typing import Generator
    if not isinstance(X, Generator):
        X = iter(X)
        Y = iter(Y)
    count = max(targetIds) - min(targetIds) + 1
            
    return [(X.next(), Y.next()) for i in count]

def genDataNonSequential(targetIds, data):
    for key in data.keys() 
    return [(X[i], Y[i]) for i in targetIds]


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
