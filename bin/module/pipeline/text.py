import pickle
import pandas as pd

import bin.module.text.Preprocessor as TextPreprocessor
import bin.module.text.Summarizer as TextSummarizer

import bin.module.util as util
from bin.setting import path

logger = util.general.initLogger(loggerName='pipeline')


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
        data.ats_normalized = TextSummarizer.Model_Sentiment.preprocess_text(util.data.DfDispatcher(path.textIMDBDf).getCol('text'), save=('imdb', True))

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
        model.sentiment.load(path.modelFolder + 'model_sentiment.pkl')
        logger.info('Loaded sentiment model with mapping.')
    else:
        model.sentiment = TextSummarizer.Model_Sentiment(mapping=data.mapping)
        model.sentiment.params.update(vocabSize=len(data.embMatrix), embWeightInit=data.embMatrix)
        model.sentiment.compile()

        #Reset training epoch tracker
        session.modelSentimentEpoch = 0


def summarize_trainSentiment(data, model, session, epochs=1):
    model.sentiment.train(data.datasets.train.x, data.datasets.train.y)
    model.sentiment.evaluate(data.datasets.test.x, data.datasets.test.y)
    model.sentiment.save(path.modelFolder + 'model_sentiment.pkl', mapping=data.mapping)
    logger.info('Saved sentiment model with mapping.')

    #Track the number of epochs trained
    session.modelSentimentEpoch += epochs
    logger.info('The sentiment model has been trained with {} epochs.'.format(session.modelSentimentEpoch))


def summarize_predictSentiment(text, model):
    # text = ['This is a test for text preprocessing. Do you think this could be a good way to expand your knowledge? Is that because theres always an inherent overhead to using classes in Python? And if so, where does the overhead come from technically speaking.']
    model.sentiment.predict(TextSummarizer.Model_Sentiment.preprocess_text(text))
