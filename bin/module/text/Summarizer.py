import pandas as pd
import numpy as np
import os
import re
from functools import partial

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




class Model_Sentiment(util.data.KerasModelBase, util.data.KerasModel):
    """
    - `preprocess_text`: text -> tokens.
    - `preprocess_token`: tokens -> tokens w format specifically for this model.
    """

    def __init__(self, mapping=object()):
        #`model` and `params` objects are created and handled in the inherited class
        super().__init__(config.modelSentimentParams)
        self.mapping = mapping

    @staticmethod
    def preprocess_text(ats, save=None):
        """
        - `ats` = a list of articles (string).
        - Provide a desc to save the normalized tokens. The desc will also be used in the filename.
        """
        ats_tokenized = TextPreprocessor.Tokenizer(ats).tokenize()
        ats_normalized = TextPreprocessor.Normalizer(ats_tokenized).lower().filterStop().filterNonWord().filter().lemmatize().getNormalized()

        if save: TextPreprocessor.saveTokens(ats_tokenized, ats_normalized, save)
        return ats_normalized

    def preprocess_tokenX(self, x):
        #Text to index and remove sentence structure
        #Use `get` return `None` when KeyError -> skipping terms not in dict
        x = [list(util.general.flattenList([[self.mapping.token2id.get(term) for term in st if self.mapping.token2id.get(term)] for st in at])) for at in x]

        #Pad sequence
        x = sequence.pad_sequences(np.array(x), **config.modelSentimentParams.config_padSequence)

        logger.info('x shape: {}'.format(x.shape))
        return x

    def preprocess_tokenY(self, y):
        y = np.array(y)

        logger.info('y shape: {}'.format(y.shape))
        return y

    def compile(self):
        """
        Must provide either `self.params.embWeightInit` or `self.params.vocabSize`.
        """
        if self.params.get('embWeightInit') is not None:
            self.params.update(vocabSize=len(self.params.embWeightInit))
        weights = self.params.get('embWeightInit', np.zeros([self.params.get('vocabSize'), self.params.embSize]))
        EmbWPresetWeight = Embedding(
            input_dim=self.params.vocabSize,
            output_dim=self.params.embSize,
            input_length=self.params.config_padSequence['maxlen'],
            weights=[weights],
            trainable=self.params.embTrainable
        )

        inputs = Input(shape=(self.params.config_padSequence['maxlen'],), dtype='int32')
        _ = EmbWPresetWeight(inputs)
        _ = Dropout(self.params.dropoutRate)(_)
        _ = Conv1D(**self.params.config_conv1D, strides=1, padding='valid', activation='relu')(_)
        _ = MaxPooling1D(self.params.poolSize)(_)
        _ = LSTM(units=self.params.LSTMUnits)(_)
        outputs = Dense(1, activation='linear')(_)

        super().compile(inputs, outputs)
        logger.info('Compiled sentiment model successfully.')

    def train(self, x_train, y_train):
        super().train(self.preprocess_tokenX(x_train), self.preprocess_tokenY(y_train))
        logger.info('-' * 60)
        logger.info('Trained with {} epochs.'.format(self.params.config_training['epochs']))

    def evaluate(self, x_test, y_test):
        loss, mae = super().evaluate(self.preprocess_tokenX(x_test), self.preprocess_tokenY(y_test))
        logger.info('-' * 60)
        logger.info('Evaluate')
        logger.info('loss value: {}'.format(loss))
        logger.info('mean absolute error: {}'.format(mae))

    def predict(self, x_new):
        prediction = super().predict(self.preprocess_tokenX(x_new))
        logger.info('-' * 60)
        logger.info('Prediction')
        logger.info('input: {}'.format(x_new))
        logger.info('output: {}'.format(prediction))




class Model_EncoderDecoder(util.data.KerasModelBase, util.data.KerasModelGen):
    #With teacher forcing, used to acquire the encoding

    def __init__(self, mapping_review=object(),  mapping_verdict=object()):
        super().__init__(config.modelEncoderDecoderParams)
        self.mapping_review = mapping_review
        self.mapping_verdict = mapping_verdict

    @staticmethod
    def preprocess_textX(ats, save=None):
        """
        - Remove punctuation (sentence structure).
        - Provide a desc to save the normalized tokens. The desc will also be used in the filename.
        """
        ats_tokenized = TextPreprocessor.Tokenizer(ats).tokenize()
        ats_normalized = TextPreprocessor.Normalizer(ats_tokenized).lower().filterStop().filterNonWord().filter().lemmatize().getNormalized()
        
        if save: TextPreprocessor.saveTokens(ats_tokenized, ats_normalized, save)
        return ats_normalized

    @staticmethod
    def preprocess_textY(ats, save=None):
        """
        - Keep punctuation (sentence structure).
        - Provide a desc to save the normalized tokens. The desc will also be used in the filename.
        """
        ats_tokenized = TextPreprocessor.Tokenizer(ats).tokenize()
        ats_normalized = TextPreprocessor.Normalizer(ats_tokenized).lower().filter().getNormalized()

        #Ensure the sentence-ending token
        for at in ats_normalized:
            for st in at:
                if st[-1] not in ['.', '!', '?']: st.append('.') 

        if save: TextPreprocessor.saveTokens(ats_tokenized, ats_normalized, save)
        return ats_normalized

    def preprocess_token(self, ats, mapping):
        #Text to index and use ats as units
        #Use `get` return `None` when KeyError -> skipping terms not in dict
        ats = [np.array(list(util.general.flattenList([[mapping.token2id.get(term) for term in st if mapping.token2id.get(term)] for st in at]))) for at in ats]

        logger.info('Shape of articles: ({}, None)'.format(len(ats)))
        return ats

    def compile(self):
        """
        Must provide either:
        - `self.params.embWeightInit_review` and `self.params.embWeightInit_verdict`
        - or `self.params.vocabSize_review` and `self.params.vocabSize_verdict`
        """
        if self.params.get('embWeightInit_review') is not None:
            self.params.update(vocabSize_review=len(self.params.embWeightInit_review))
        if self.params.get('embWeightInit_verdict') is not None:
            self.params.update(vocabSize_verdict=len(self.params.embWeightInit_verdict))
        weights_review = self.params.get('embWeightInit_review', np.zeros([self.params.get('vocabSize_review'), self.params.encoderEmb['size']]))
        weights_verdict = self.params.get('embWeightInit_verdict', np.zeros([self.params.get('vocabSize_verdict'), self.params.decoderEmb['size']]))
        Emb_Encoder = Embedding(
            input_dim=self.params.vocabSize_review,
            output_dim=self.params.encoderEmb['size'],
            weights=[weights_review],
            trainable=self.params.encoderEmb['trainable']
        )
        Emb_Decoder = Embedding(
            input_dim=self.params.vocabSize_verdict,
            output_dim=self.params.decoderEmb['size'],
            weights=[weights_verdict],
            trainable=self.params.decoderEmb['trainable']
        )

        #Input the whole text
        inputs_encoder = Input(shape=(None,), dtype='int32', name='inputs_encoder')
        _ = Emb_Encoder(inputs_encoder)
        _ = Dropout(self.params.dropoutRate)(_)
        _ = Conv1D(**self.params.config_conv1D, strides=1, padding='valid', activation='relu')(_)
        _ = MaxPooling1D(self.params.poolSize)(_)
        _, state_h, state_c = LSTM(units=self.params.LSTMUnits, return_state=True)(_)

        #Input the verdict (t-1) for teacher forcing with the initial states from the encoder
        inputs_decoder = Input(shape=(None,), dtype='int32', name='inputs_decoder')
        _ = Emb_Decoder(inputs_decoder)
        _ = LSTM(units=self.params.LSTMUnits, return_sequences=True)(_, initial_state=[state_h, state_c])

        #Output the verdict
        outputs = Dense(len(weights_verdict), activation='softmax', name='outputs')(_)

        super().compile([inputs_encoder, inputs_decoder], outputs)
        logger.info('Compiled encoder-decoder model successfully.')

    def _genData(self, targetIds):
        targetId = targetIds[0]

        #Get onehot and reshape np arrays
        xEncoder_train = self.xEncoder_train[targetId].reshape(1, len(self.xEncoder_train[targetId]))
        xDecoder_train = self.xDecoder_train[targetId].reshape(1, len(self.xDecoder_train[targetId]))
        y_onehot = util.data.ids2Onehot(self.y_train[targetId], self.params.vocabSize_verdict)
        y_onehot = y_onehot.reshape(1, *y_onehot.shape)

        return (
            {'inputs_encoder': xEncoder_train,
            'inputs_decoder': xDecoder_train},
            {'outputs': y_onehot}
        )

    def train(self, review, verdict):
        #Process tokens, shift each `at` for teacher forcing
        verdict_processed = self.preprocess_token(verdict, self.mapping_verdict)
        self.xEncoder_train = self.preprocess_token(review, self.mapping_review)
        self.xDecoder_train = [at[:-1] for at in verdict_processed]
        self.y_train = [at[1:] for at in verdict_processed]

        #Prepare data dispatcher
        dataDispatcher = util.data.KerasDataDispatcher(
            sampleSize=len(verdict_processed),
            batchSize=1, #Batch size must be 1 while the articles have dif len (if we decide not to pad)
            genData=self._genData
        )

        #Training
        super().train(dataDispatcher)
        logger.info('-' * 60)
        logger.info('Trained with {} epochs.'.format(self.params.config_training['epochs']))

    def evaluate(self): pass
        #No need to evaluate. The real outputs are those of the inference model which uses the encoding from this model as input

    def predict(self): pass
        #Due to the extra input of teacher forcing, can't be directly used to predict -> Use the decoder models