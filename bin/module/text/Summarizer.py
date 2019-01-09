import pandas as pd
import os
import re

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, MaxPooling1D

import bin.module.util as util
from bin.setting import path, textSummarizer as config

logger = util.general.initLogger(loggerName='TextSummarizer')




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

        print('Read in {} reviews.'.format(df.shape[0]))
        return df
    
    @classmethod
    def exportDf(cls):
        cls.readAsDf().to_csv(path.textIMDBDf, encoding='utf-8')
        print('Export the df at {}.'.format(path.textIMDBDf))




class Model_Sentiment():

    def __init__(self, data, model=object()):
        self.data = data

        try:
            self.data.train.x, self.data.train.y
            self.data.test.x, self.data.test.y
        except AttributeError as e:
            raise AttributeError('Data object must conform to the predefined structure. Refer to the class definition.') from e
            
        self.model = model
        self.params = config.modelSentimentParams #Default

    def preprocess(self):
        #Pad sequence
        self.data.train.x = sequence.pad_sequences(self.data.train.x, **self.params.config_padSequence)
        self.data.test.x = sequence.pad_sequences(self.data.test.x, **self.params.config_padSequence)
        print('x_train shape:', self.data.train.x.shape)
        print('x_test shape:', self.data.test.x.shape)

    def compile(self):
        EmbWPresetWeight = Embedding(input_dim=self.params.vocabSize, output_dim=300)
        if self.params.embWeightInit:
            EmbWPresetWeight.set_weights(self.params.embWeightInit)

        inputs = Input(shape=(self.params.config_padSequence['maxlen'], ), dtype='int32')
        _ = EmbWPresetWeight(inputs)
        _ = Dropout(self.params.dropoutRate)(_)
        _ = Conv1D(**self.params.config_conv1D, strides=1, padding='valid', activation='relu')(_)
        _ = MaxPooling1D(self.params.poolSize)(_)
        _ = LSTM(**self.params.config_LSTM)(_)
        outputs = Dense(1, activation='linear')(_)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='logcosh', optimizer='adam', metrics=['logcosh']) #TODO: customize the metric
        self.model.summary()

    def train(self):
        self.model.fit(self.data.train.x, self.data.train.y, batchSize=self.params.batchSize, epochs=self.params.epochs)

    def evaluate(self):
        score, acc = self.model.evaluate(self.data.test.x, self.data.test.y, batchSize=self.params.batchSize)
        print('Test score:', score)
        print('Test accuracy:', acc)
    
    def predict(self, newX):
        prediction = self.model.predict(newX, batchSize=self.params.batchSize)
        print('Prediction')
        print('input:', newX)
        print('output:', prediction)

    def save(self):
        model.get_config()
        model = Model.from_config(config)

        model.get_weights()
        model.set_weights(weights)




import bin.module.text.Preprocessor as TextPreprocessor
import pickle

df = pd.read_csv(path.textIMDBDf)
# tokenizer = TextPreprocessor.Tokenizer(util.data.DfDispatcher(path.textIMDBDf).getCol('text'))
# ats_tokenized = tokenizer.tokenize()
# with open(path.textTkFolder + 'tokenized_imdb.pkl', 'wb') as f:
#     pickle.dump(ats_tokenized, f)
with open(path.textTkFolder + 'tokenized_imdb.pkl', 'rb') as f:
    ats_tokenized = pickle.load(f)

# normalizer = TextPreprocessor.Normalizer(ats_tokenized)
# ats_normalized = normalizer.lower().filterStop().filterNonWord().filter().getNormalized()
# with open(path.textTkFolder + 'normalized_imdb.pkl', 'wb') as f:
#     pickle.dump(ats_normalized, f)
with open(path.textTkFolder + 'normalized_imdb.pkl', 'rb') as f:
    ats_normalized = pickle.load(f)

nSample = len(ats_normalized)
(id_train, id_test), _ = util.data.SetDivider(proportion=[0.8, 0.2], nSample=nSample).divideSets()

data = util.general.DataContainer({
    'train': {
        'x': [ats_normalized[id] for id in id_train],
        'y': [df.iloc[id].rating for id in id_train]
    },
    'test': {
        'x': [ats_normalized[id] for id in id_test],
        'y': [df.iloc[id].rating for id in id_test]
    }
})
emb = TextPreprocessor.EmbOperation.loadPretrainedEmb(path.gNewsW2V)

model = Model_Sentiment(data)
model.params.embWeightInit = ''
model.preprocess()
model.compile()
model.train()
model.evaluate()
model.predict("")