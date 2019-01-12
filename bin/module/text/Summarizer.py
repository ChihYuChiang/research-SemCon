import pandas as pd
import numpy as np
import os
import re

from keras.preprocessing import sequence
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

        logger.info('Read in {} reviews.'.format(df.shape[0]))
        return df
    
    @classmethod
    def exportDf(cls):
        cls.readAsDf().to_csv(path.textIMDBDf, encoding='utf-8')
        logger.info('Export the df at {}.'.format(path.textIMDBDf))




class Model_Sentiment(util.data.KerasModel):

    def __init__(self, params={}, mapping=object()):
        #`model` and `params` objects are created and handled in the inherited class
        super().__init__(params)
        self.mapping = mapping

    def preprocess_x(self, x):
        #Text to index and remove sentence structure
        x = [list(util.general.flattenList([[self.mapping.token2id[term] for term in st] for st in at])) for at in x]

        #Pad sequence
        x = sequence.pad_sequences(np.array(x), **config.modelSentimentParams.config_padSequence)

        logger.info('x shape: {}'.format(x.shape))
        return x

    def preprocess_y(self, y):
        y = np.array(y)

        logger.info('y shape: {}'.format(y.shape))
        return y

    def compile(self):
        weights = self.params.embWeightInit if self.params.embWeightInit is not None else np.zeros([self.params.vocabSize, self.params.embSize])
        EmbWPresetWeight = Embedding(
            input_dim=self.params.vocabSize, output_dim=self.params.embSize,
            input_length=self.params.config_padSequence['maxlen'],
            weights=[weights], trainable=True
        )

        inputs = Input(shape=(self.params.config_padSequence['maxlen'], ), dtype='int32')
        _ = EmbWPresetWeight(inputs)
        _ = Dropout(self.params.dropoutRate)(_)
        _ = Conv1D(**self.params.config_conv1D, strides=1, padding='valid', activation='relu')(_)
        _ = MaxPooling1D(self.params.poolSize)(_)
        _ = LSTM(**self.params.config_LSTM)(_)
        outputs = Dense(1, activation='linear')(_)
        super().compile(inputs, outputs)

    def train(self, x_train, y_train):
        super().train(self.preprocess_x(x_train), self.preprocess_y(y_train))
        logger.info('-' * 60)
        logger.info('Trained with {} epochs'.format(self.params.config_training['epochs']))

    def evaluate(self, x_test, y_test):
        loss, mae = super().evaluate(self.preprocess_x(x_test), self.preprocess_y(y_test))
        logger.info('-' * 60)
        logger.info('Evaluate')
        logger.info('loss value: {}'.format(loss))
        logger.info('mean absolute error: {}'.format(mae))

    def predict(self, x_new):
        prediction = super().predict(self.preprocess_x(x_new))
        logger.info('-' * 60)
        logger.info('Prediction')
        logger.info('input: {}'.format(x_new))
        logger.info('output: {}'.format(prediction))


import bin.module.text.Preprocessor as TextPreprocessor
import pickle

df = pd.read_csv(path.textIMDBDf)
# ats_tokenized = TextPreprocessor.Tokenizer(util.data.DfDispatcher(path.textIMDBDf).getCol('text')).tokenize()
# with open(path.textTkFolder + 'tokenized_imdb.pkl', 'wb') as f:
#     pickle.dump(ats_tokenized, f)
# with open(path.textTkFolder + 'tokenized_imdb.pkl', 'rb') as f:
#     ats_tokenized = pickle.load(f)

# ats_normalized = TextPreprocessor.Normalizer(ats_tokenized).lower().filterStop().filterNonWord().filter().getNormalized()
# with open(path.textTkFolder + 'normalized_imdb.pkl', 'wb') as f:
#     pickle.dump(ats_normalized, f)
with open(path.textTkFolder + 'normalized_imdb.pkl', 'rb') as f:
    ats_normalized = pickle.load(f)

# mapper = TextPreprocessor.Mapping(ats_normalized)
# mapper.brief()
# mapper.makeDict()
# mapper.dict.save(path.textDictFolder + 'mapping_imdb.pkl')
mapping = TextPreprocessor.Mapping().dict.load(path.textDictFolder + 'mapping_imdb.pkl')
sortedVocab = [mapping.id2token[i] for i in range(max(mapping.keys()) + 1)]

# emb = TextPreprocessor.EmbOperation.loadPretrainedEmb(path.gNewsW2V)
# embMatrix = TextPreprocessor.EmbOperation.sliceEmb(sortedVocab, emb)
# with open(path.textEmbFolder + 'emb_imdb_raw.pkl', 'wb') as f:
#     pickle.dump(embMatrix, f)
with open(path.textEmbFolder + 'emb_imdb_raw.pkl', 'rb') as f:
    embMatrix = pickle.load(f)

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

model = Model_Sentiment(params=config.modelSentimentParams, mapping=mapping)
model.params.update(vocabSize=len(embMatrix), embWeightInit=embMatrix)

model.compile()
model.train(data.train.x, data.train.y)
model.evaluate(data.test.x, data.test.y)
model.save(path.modelFolder + 'model_sentiment.pkl', mapping=mapping)

model = Model_Sentiment()
model.load(path.modelFolder + 'model_sentiment.pkl')

model.predict(data.new.x)


['This is a test for text preprocessing. Do you think this could be a good way to expand your knowledge?', 'Is that because theres always an inherent overhead to using classes in Python? And if so, where does the overhead come from technically speaking.', ' ']
new_tokenized = TextPreprocessor.Tokenizer(x).tokenize()
new_normalized = TextPreprocessor.Normalizer(new_tokenized).lower().filterStop().filterNonWord().filter().getNormalized()
