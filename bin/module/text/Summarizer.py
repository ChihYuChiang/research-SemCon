import pandas as pd
import numpy as np
import os
import re

from keras.preprocessing import sequence
from keras.layers import Input, Dense, Dropout
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, MaxPooling1D

import bin.module.util as util
import bin.module.text.Preprocessor as TextPreprocessor
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

    def __init__(self, mapping=object()):
        #`model` and `params` objects are created and handled in the inherited class
        super().__init__(config.modelSentimentParams)
        self.mapping = mapping

    @staticmethod
    def preprocess_text(ats, save=('', False)):
        """
        `ats` = a list of articles (string).
        """
        ats_tokenized = TextPreprocessor.Tokenizer(ats).tokenize()
        ats_normalized = TextPreprocessor.Normalizer(ats_tokenized).lower().filterStop().filterNonWord().filter().getNormalized()

        if save[1]:
            with open('{}tokenized_{}.pkl'.format(path.textTkFolder, save[0]), 'wb') as f: pickle.dump(ats_tokenized, f)
            with open('{}normalized_{}.pkl'.format(path.textTkFolder, save[0]), 'wb') as f: pickle.dump(ats_normalized, f)
            logger.info('Saved tokenized and normalized {} text at {}.'.format(save[0], path.textTkFolder))            
        
        return ats_normalized

    def preprocess_x(self, x):
        #Text to index and remove sentence structure
        #Use `get` return `None` when KeyError -> skipping terms not in dict
        x = [list(util.general.flattenList([[self.mapping.token2id.get(term) for term in st if self.mapping.token2id.get(term)] for st in at])) for at in x]

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
        logger.info('Compiled sentiment model successfully.')

    def train(self, x_train, y_train):
        super().train(self.preprocess_x(x_train), self.preprocess_y(y_train))
        logger.info('-' * 60)
        logger.info('Trained with {} epochs.'.format(self.params.config_training['epochs']))

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




class Model_EncoderDecoder(util.data.KerasModel):
    #With teacher forcing, used to acquire the encoding

    def __init__(self, mapping=object()):
        super().__init__(config.modelEncoderDecoderParams)
        self.mapping = mapping

    @staticmethod
    def preprocess_text(ats, save=('', False)):
        """
        `ats` = a list of articles (string).
        """
        ats_tokenized = TextPreprocessor.Tokenizer(ats).tokenize()
        ats_normalized = TextPreprocessor.Normalizer(ats_tokenized).lower().filterStop().filterNonWord().filter().getNormalized()

        if save[1]:
            with open('{}tokenized_{}.pkl'.format(path.textTkFolder, save[0]), 'wb') as f: pickle.dump(ats_tokenized, f)
            with open('{}normalized_{}.pkl'.format(path.textTkFolder, save[0]), 'wb') as f: pickle.dump(ats_normalized, f)
            logger.info('Saved tokenized and normalized {} text at {}.'.format(save[0], path.textTkFolder))            
        
        return ats_normalized

    def preprocess_x(self, x):
        #Text to index and remove sentence structure
        #Use `get` return `None` when KeyError -> skipping terms not in dict
        x = [list(util.general.flattenList([[self.mapping.token2id.get(term) for term in st if self.mapping.token2id.get(term)] for st in at])) for at in x]

        #Pad sequence
        x = sequence.pad_sequences(np.array(x), **config.modelSentimentParams.config_padSequence)

        logger.info('x shape: {}'.format(x.shape))
        return x

    def preprocess_y(self, y):
        pass

    def compile(self):
        weights = self.params.embWeightInit if self.params.embWeightInit is not None else np.zeros([self.params.vocabSize, self.params.embSize])
        EmbWPresetWeight = Embedding(
            input_dim=self.params.vocabSize, output_dim=self.params.embSize,
            input_length=self.params.config_padSequence['maxlen'],
            weights=[weights], trainable=True
        )

        #Input the whole text
        inputs_encoder = Input(shape=(self.params.config_padSequence['maxlen'], ), dtype='int32')
        _ = EmbWPresetWeight(inputs_encoder)
        _ = Dropout(self.params.dropoutRate)(_)
        _ = Conv1D(**self.params.config_conv1D, strides=1, padding='valid', activation='relu')(_)
        _ = MaxPooling1D(self.params.poolSize)(_)
        _, state_h, state_c = LSTM(**self.params.config_encoderLSTM)(_)

        #Input the verdict (t-1) for teacher forcing with the initial states from the encoder
        inputs_decoder = Input(shape=(None, num_decoder_tokens))
        _ = LSTM(**self.params.config_decoderLSTM)(inputs_decoder, initial_state=[state_h, state_c])

        #Output the verdict
        outputs = Dense(num_decoder_tokens, activation='softmax')(_)

        super().compile([inputs_encoder, inputs_decoder], outputs)
        logger.info('Compiled encoder model successfully.')

    def train(self, x_train, y_train):
        super().train(self.preprocess_x(x_train), self.preprocess_y(y_train))
        logger.info('-' * 60)
        logger.info('Trained with {} epochs.'.format(self.params.config_training['epochs']))

    def evaluate(self): pass
        #No need to evaluate. The real outputs are those of the inference model which uses the encoding from this model as input

    def predict(self): pass
        #Due to the extra input of teacher forcing, can't be directly used to predict -> Use the decoder models