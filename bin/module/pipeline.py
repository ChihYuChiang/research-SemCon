import pickle
from os import listdir

import bin.module.ImgDownloader as ImgDownloader
# import bin.module.ImgPreprocessor as ImgPreprocessor
# import bin.module.TextPreprocessor as TextPreprocessor
# import bin.module.TextSummarizer as TextSummarizer

import bin.module.util as util
from bin.setting import path


#--Initialization
#Provide the attributes to be overwritten in an obj
def initialize(sessionPath, overwrite=False):
    data = util.UniversalContainer()

    #Load id and game title mapping
    with open(path.mapping, 'rb') as f: data.mapping = pickle.load(f)
    
    #Load session. If no file found, create new session with the template
    newSession = {
        'currentDownloadId': 0,
        'currentSearchId': 0
    }
    session = util.Session.load(sessionPath, **newSession)
    
    return data, session


#--Observe session outcome
def observeOutcome(data, session):
    print('-' * 60)
    print('session')
    print(session)
    print('-' * 60)
    print('data')
    print(data)
    print('-' * 60)


#--Search image
def imgDownload_search(data, session, batchSize=10):
    #Perform search
    data.responses, session.currentSearchId = ImgDownloader.Searcher.searchBatch(data.mapping, startId=session.currentSearchId, batchSize=batchSize)

    #Save search responses to multiple files
    util.writeJsls(data.responses, '{}{}.jsl'.format(path.imageResFolder, session.currentSearchId))


#--Parse and consolidate response
def imgDownload_parse(data):
    with open(path.imageUrl, 'wb') as f:
        data.urlInfo = []

        for p in listdir(path.imageResFolder):
            #Load search responses from file
            data.responses = util.readJsls(path.imageResFolder + p)

            #Parse responses for url info
            data.urlInfo.extend(ImgDownloader.Searcher.parseResponse_n(data.responses))

        #Save url info to file
        pickle.dump(data.urlInfo, f)
        print('Pickled {} items\' urls.'.format(len(data.urlInfo)))


#--Download image
def imgDownload_download(data, session, batchSize=3, urlIdRange=False):
    #Load url info from file
    with open(path.imageUrl, 'rb') as f: data.urlInfo = pickle.load(f)

    #Perform download
    session.currentDownloadId, session.failedUrl = ImgDownloader.Downloader.download8SaveBatch(data.urlInfo, startId=session.currentDownloadId, batchSize=batchSize, urlIdRange=urlIdRange)


#--Identify failed items from img result
#Only use it when the session failed
def imgDownload_identifyFailures(session, lastTargetId, urlIdRange=False):
    session.currentDownloadId, session.failedUrl = ImgDownloader.Downloader.identifyFailures(lastTargetId, urlIdRange)


#--Download failed image
def imgDownload_reDownload(data, session):
    #Load url info from file
    with open(path.imageUrl, 'rb') as f: data.urlInfo = pickle.load(f)
    
    #Perform redownload and update the failed list
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