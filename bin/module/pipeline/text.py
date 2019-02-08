import pickle
import pandas as pd
from tensorflow.python.framework.errors_impl import UnimplementedError

import bin.module.text.Preprocessor as TextPreprocessor
import bin.module.text.Summarizer as TextSummarizer

import bin.module.util as util
from bin.setting import path, cred

logger = util.general.initLogger(
    loggerName='pipeline',
    slack={
        **cred.Slack,
        'level': 'WARNING',
        'channel': '#py-logger'
    })




'''
------------------------------------------------------------
Sentiment Model
------------------------------------------------------------
'''
def preprocess_initSentiment(data, load=True):
    #Raw
    data.df = pd.read_csv(path.textIMDBDf)

    if load: #Load precomputed data objects
        with open('{}normalized_imdb.pkl'.format(path.textTkFolder), 'rb') as f:
            data.ats_normalized = pickle.load(f)
        data.mapping = TextPreprocessor.Mapping().dict.load(path.textDictFolder + 'mapping_imdb.pkl')
        with open(path.textEmbFolder + 'emb_imdb_raw.pkl', 'rb') as f:
            data.embMatrix = pickle.load(f)
    else:
        #Token and normalization
        data.ats_normalized = TextSummarizer.Model_Sentiment.preprocess_text(util.data.DfDispatcher(path.textIMDBDf).getCol('text'), save='imdb')

        #Vocab dictionary
        mapper = TextPreprocessor.Mapping(data.ats_normalized)
        mapper.brief()
        mapper.makeDict()
        mapper.dict.save(path.textDictFolder + 'mapping_imdb.pkl')
        data.mapping = mapper.dict

        #Embedding
        sortedVocab = [data.mapping.id2token[i] for i in range(max(data.mapping.keys()) + 1)]
        emb = TextPreprocessor.EmbOperation.loadPretrainedEmb(path.gNewsW2V)
        data.embMatrix = TextPreprocessor.EmbOperation.sliceEmb(sortedVocab, emb)
        with open(path.textEmbFolder + 'emb_imdb_raw.pkl', 'wb') as f:
            pickle.dump(data.embMatrix, f)

    #Input datasets
    (data.id_train, data.id_test), _ = util.data.SetDivider(proportion=[0.8, 0.2], nSample=len(data.ats_normalized)).divideSets()
    
    data.datasets = util.general.DataContainer({
        'train': {
            'x': [data.ats_normalized[id] for id in data.id_train],
            'y': [data.df.iloc[id].rating for id in data.id_train]
        },
        'test': {
            'x': [data.ats_normalized[id] for id in data.id_test],
            'y': [data.df.iloc[id].rating for id in data.id_test]
        }
    })

    logger.info('Processed data for sentiment model.')


def summarize_initSentiment(data, model, session, load=True):
    if load:
        model.sentiment = TextSummarizer.Model_Sentiment()
        model.sentiment.load(path.modelFolder + 'model_sentiment/')
        logger.info('Loaded sentiment model with mapping.')
    else:
        model.sentiment = TextSummarizer.Model_Sentiment(mapping=data.mapping)
        model.sentiment.params.update(embWeightInit=data.embMatrix)
        model.sentiment.compile()

        #Reset training epoch tracker
        session.modelSentimentEpoch = 0


def summarize_trainSentiment(data, model, session, epochs=1):
    model.sentiment.params.config_training['epochs'] = epochs

    model.sentiment.train(data.datasets.train.x, data.datasets.train.y)
    model.sentiment.evaluate(data.datasets.test.x, data.datasets.test.y)
    model.sentiment.save(path.modelFolder + 'model_sentiment/', mapping=data.mapping)
    logger.info('Saved sentiment model with mapping.')

    #Track the number of epochs trained
    session.modelSentimentEpoch += epochs
    logger.info('The sentiment model has been trained with {} epochs.'.format(session.modelSentimentEpoch))


def summarize_predictSentiment(text, model):
    # text = ['This is a test for text preprocessing. Do you think this could be a good way to expand your knowledge? Is that because theres always an inherent overhead to using classes in Python? And if so, where does the overhead come from technically speaking.']
    model.sentiment.predict(TextSummarizer.Model_Sentiment.preprocess_text(text))




'''
------------------------------------------------------------
Encoder-Decoder Model
------------------------------------------------------------
'''
def preprocess_initEncoderDecoder(data, load=True):
    #Raw
    data.df = pd.read_csv(path.textDf, encoding='cp1252')

    if load: #Load precomputed data objects
        with open('{}normalized_review_gs.pkl'.format(path.textTkFolder), 'rb') as f:
            data.review_normalized = pickle.load(f)
        with open('{}normalized_verdict.pkl'.format(path.textTkFolder), 'rb') as f:
            data.verdict_normalized = pickle.load(f)
        data.mapping_review = TextPreprocessor.Mapping().dict.load(path.textDictFolder + 'mapping_review_gs.pkl')
        data.mapping_verdict = TextPreprocessor.Mapping().dict.load(path.textDictFolder + 'mapping_verdict.pkl')
        with open(path.textEmbFolder + 'emb_review_gs_raw.pkl', 'rb') as f:
            data.embMatrix_review = pickle.load(f)
        with open(path.textEmbFolder + 'emb_verdict_raw.pkl', 'rb') as f:
            data.embMatrix_verdict = pickle.load(f)

    else:
        #Token and normalization
        config_dfDispatcher = {
            'filePath': path.textDf,
            'encoding': 'cp1252',
            'targetCols': ['Review', 'Short Description']
        }
        data.review_normalized = TextSummarizer.Model_EncoderDecoder.preprocess_textX(util.data.DfDispatcher(**config_dfDispatcher).getCol('Review'), save='review_gs')
        data.verdict_normalized = TextSummarizer.Model_EncoderDecoder.preprocess_textY(util.data.DfDispatcher(**config_dfDispatcher).getCol('Short Description'), save='verdict')

        #Vocab dictionary
        mapper_review = TextPreprocessor.Mapping(data.review_normalized)
        mapper_review.brief()
        mapper_review.makeDict()
        mapper_review.dict.save(path.textDictFolder + 'mapping_review_gs.pkl')
        data.mapping_review = mapper_review.dict

        mapper_verdict = TextPreprocessor.Mapping(data.verdict_normalized)
        mapper_verdict.brief()
        mapper_verdict.makeDict()
        mapper_verdict.dict.save(path.textDictFolder + 'mapping_verdict.pkl')
        data.mapping_verdict = mapper_verdict.dict

        #Embedding
        emb = TextPreprocessor.EmbOperation.loadPretrainedEmb(path.gNewsW2V)

        sortedVocab_review = [data.mapping_review.id2token[i] for i in range(max(data.mapping_review.keys()) + 1)]
        data.embMatrix_review = TextPreprocessor.EmbOperation.sliceEmb(sortedVocab_review, emb)
        with open(path.textEmbFolder + 'emb_review_gs_raw.pkl', 'wb') as f:
            pickle.dump(data.embMatrix_review, f)

        sortedVocab_verdict = [data.mapping_verdict.id2token[i] for i in range(max(data.mapping_verdict.keys()) + 1)]
        data.embMatrix_verdict = TextPreprocessor.EmbOperation.sliceEmb(sortedVocab_verdict, emb)
        with open(path.textEmbFolder + 'emb_verdict_raw.pkl', 'wb') as f:
            pickle.dump(data.embMatrix_verdict, f)

    logger.info('Processed data for encoder-decoder model.')


def summarize_initEncoderDecoder(data, model, session, load=True):
    if load:
        model.encoderDecoder = TextSummarizer.Model_EncoderDecoder()
        model.encoderDecoder.load(path.modelFolder + 'model_encoder-decoder/')
        logger.info('Loaded encoder-decoder model with mappings.')
    else:
        model.encoderDecoder = TextSummarizer.Model_EncoderDecoder(mapping_review=data.mapping_review, mapping_verdict=data.mapping_verdict)
        model.encoderDecoder.params.update(embWeightInit_review=data.embMatrix_review, embWeightInit_verdict=data.embMatrix_verdict)
        model.encoderDecoder.compile()

        #Reset training epoch tracker
        session.modelEncoderDecoderEpoch = 0


def summarize_trainEncoderDecoder(data, model, session, epochs=1):
    model.encoderDecoder.params.config_training['epochs'] = epochs

    complete = True #Deal with Keras' internal error when multi-processing
    try:
        model.encoderDecoder.train(data.review_normalized, data.verdict_normalized)
    except UnimplementedError:
        logger.warning('Training stopped due to a Keras\' internal error (the session tracker will not be updated).')
        complete = False

    model.encoderDecoder.save(path.modelFolder + 'model_encoder-decoder/', mapping_review=data.mapping_review, mapping_verdict=data.mapping_verdict)
    logger.info('Saved encoder-decoder model with mappings.')

    if complete:
        #Track the number of epochs trained
        session.modelEncoderDecoderEpoch += epochs
        logger.info('The encoder-decoder model has been trained with {} epochs.'.format(session.modelEncoderDecoderEpoch))


def summarize_predictEncoderDecoder(text, model):
    # text = ['This is a test for text preprocessing. Do you think this could be a good way to expand your knowledge? Is that because theres always an inherent overhead to using classes in Python? And if so, where does the overhead come from technically speaking.']
    model.encoderDecoder.predict(TextSummarizer.Model_EncoderDecoder.preprocess_textX(text))