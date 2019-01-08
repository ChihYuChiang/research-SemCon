import pickle
from os import listdir

import bin.module.img.Downloader as ImgDownloader
# import bin.module.img.Preprocessor as ImgPreprocessor
import bin.module.text.Preprocessor as TextPreprocessor
# import bin.module.text.Summarizer as TextSummarizer

import bin.module.util as util
from bin.setting import path

logger = util.general.initLogger(loggerName='pipeline')


#--Initialization
#Provide the attributes to be overwritten in an obj
def initialize(sessionPath, overwrite=False):
    data = util.general.UniversalContainer()

    try:
        #Load id and game title mapping
        with open(path.mapping, 'rb') as f: data.mapping = pickle.load(f)

        #Load url info from file
        with open(path.imageUrl, 'rb') as f: data.urlInfo = pickle.load(f)
    except OSError: logger.warning('Either `mapping` or `urlInfo` file not found. This can cause error in later operations.')
    
    #Load session. If no file found, create new session with the template
    newSession = {
        'currentDownloadId': 0,
        'currentSearchId': 0
    }
    session = util.general.Session.load(sessionPath, **newSession)
    
    return data, session


#TODO: Make it a decorator
#--Observe session outcome
def observeOutcome(data, session):
    logger.info('-' * 60)
    logger.info('session')
    logger.info(session)
    logger.info('-' * 60)
    logger.info('data')
    logger.info(data)
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


def textPreprocess(): pass
    #sentence vec
    #tokenized main, with exp only, remove stop, remove nonword
    #tokenized comment, with exp + stop, keep nonword
    #Tokenized comment, with exp only
    #List of exp word
    #emb - word 2 way table


def imgPreprocess(): pass
    #Cropping to be square
    #Scaling to 100px by 100px
    #Img selection: separate gameplay, logo?
    #Mean, standard deviation of input pixel
    #Normalizing
    #Augmentation: Perturbation, rotation
    #Detect anomaly image (not from the same game) 