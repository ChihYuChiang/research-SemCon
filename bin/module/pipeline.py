import pickle
import pandas as pd
from os import listdir

import bin.module.img.Downloader as ImgDownloader
# import bin.module.img.Preprocessor as ImgPreprocessor
import bin.module.text.Preprocessor as TextPreprocessor
import bin.module.text.Summarizer as TextSummarizer

import bin.module.util as util
from bin.setting import path

logger = util.general.initLogger(loggerName='pipeline')


#--Initialization
#Provide the attributes to be overwritten in an obj
def initialize(sessionPath, overwrite=False):
    data = util.general.UniversalContainer()
    model = util.general.UniversalContainer()

    try:
        #Load id and game title mapping
        with open(path.mapping, 'rb') as f: data.mapping = pickle.load(f)

        #Load url info from file
        with open(path.imageUrl, 'rb') as f: data.urlInfo = pickle.load(f)
    except OSError: logger.warning('Either `mapping` or `urlInfo` file not found. This can cause error in later operations.')
    
    #Load session. If no file found, create new session with the template
    newSession = {
        'currentDownloadId': 0,
        'currentSearchId': 0,
        'modelSentimentEpoch': 0
    }
    session = util.general.Session.load(sessionPath, **newSession)
    
    return data, model, session


#TODO: Make it a decorator
#--Observe session outcome
def observeOutcome(data, model, session):
    logger.info('-' * 60)
    logger.info('session')
    logger.info(session)
    logger.info('-' * 60)
    logger.info('data')
    logger.info(data)
    logger.info('-' * 60)
    logger.info('model')
    logger.info(model)
    logger.info('-' * 60)


#--Search image
def imgDownload_search(data, session, batchSize=10):
    #Perform search
    data.responses, session.currentSearchId = ImgDownloader.Searcher.searchBatch(data.mapping, startId=session.currentSearchId, batchSize=batchSize)

    #Save search responses to multiple files
    util.general.writeJsls(data.responses, '{}{}.jsl'.format(path.imageResFolder, session.currentSearchId))


#--Parse and consolidate response
def imgDownload_parse(data):
    with open(path.imageUrl, 'wb') as f:
        data.urlInfo = []

        for p in listdir(path.dataLake.imageResFolder):
            #Load search responses from file
            data.responses = util.general.readJsls(path.dataLake.imageResFolder + p)

            #Parse responses for url info
            data.urlInfo.extend(ImgDownloader.Searcher.parseResponseBatch(data.responses))

        #Save url info to file
        pickle.dump(data.urlInfo, f)
        logger.info('Pickled {} items\' urls.'.format(len(data.urlInfo)))


#--Download image
def imgDownload_download(data, session, batchSize=3, urlIdRange=False):
    #Perform download
    session.currentDownloadId, newlyFailedUrl = ImgDownloader.Downloader.download8SaveBatch(data.urlInfo, startId=session.currentDownloadId, batchSize=batchSize, urlIdRange=urlIdRange)

    #Expand failedUrl
    session.failedUrl.extend(newlyFailedUrl)


#--Identify failed and corrupted items from img result
#Only use it when the session failed
def imgDownload_identifyFailure8Corruption(data, session, lastTargetId=11857, urlIdRange=False):
    session.corruptedUrl = ImgDownloader.Downloader.identifyCorruptions() #Examine the HHD as external is too slow
    session.currentDownloadId, session.failedUrl = ImgDownloader.Downloader.identifyFailures(data.urlInfo, lastTargetId, urlIdRange) #Examine the external


#--Download failed and corrupted image
def imgDownload_reDownload(data, session):
    #Perform redownload and update the failed list
    session.corruptedUrl = ImgDownloader.Downloader.download8SaveFailed(data.urlInfo, session.corruptedUrl)
    session.failedUrl = ImgDownloader.Downloader.download8SaveFailed(data.urlInfo, session.failedUrl)


def textPreprocess_initSentiment(data, load=True):
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
        data.ats_normalized = Model_Sentiment.preprocess_text(util.data.DfDispatcher(path.textIMDBDf).getCol('text'), save=('imdb', True))

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
    data.nSample = len(data.ats_normalized)
    (data.id_train, data.id_test), _ = util.data.SetDivider(proportion=[0.8, 0.2], nSample=data.nSample).divideSets()
    
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


def textSummarize_initSentiment(data, model, session, load=True):
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


def textSummarize_trainSentiment(data, model, session, epochs=1):
    model.sentiment.train(data.datasets.train.x, data.datasets.train.y)
    model.sentiment.evaluate(data.datasets.test.x, data.datasets.test.y)
    model.sentiment.save(path.modelFolder + 'model_sentiment.pkl', mapping=data.mapping)
    logger.info('Saved sentiment model with mapping.')

    #Track the number of epochs trained
    session.modelSentimentEpoch += epochs
    logger.info('The sentiment model has been trained with {} epochs.'.format(session.modelSentimentEpoch))


def textSummarize_predictSentiment(text, model):
    # text = ['This is a test for text preprocessing. Do you think this could be a good way to expand your knowledge? Is that because theres always an inherent overhead to using classes in Python? And if so, where does the overhead come from technically speaking.']
    model.sentiment.predict(TextSummarizer.Model_Sentiment.preprocess_text(text))
