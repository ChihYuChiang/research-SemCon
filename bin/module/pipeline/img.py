import pickle
from os import listdir

import bin.module.img.Downloader as ImgDownloader

import bin.module.util as util
from bin.setting import path

logger = util.general.initLogger(loggerName='pipeline')


#--Search image
def download_search(data, session, batchSize=10):
    #Perform search
    data.responses, session.currentSearchId = ImgDownloader.Searcher.searchBatch(data.mapping, startId=session.currentSearchId, batchSize=batchSize)

    #Save search responses to multiple files
    util.general.writeJsls(data.responses, '{}{}.jsl'.format(path.imageResFolder, session.currentSearchId))


#--Parse and consolidate response
def download_parse(data):
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
def download_download(data, session, batchSize=3, urlIdRange=False):
    #Perform download
    session.currentDownloadId, newlyFailedUrl = ImgDownloader.Downloader.download8SaveBatch(data.urlInfo, startId=session.currentDownloadId, batchSize=batchSize, urlIdRange=urlIdRange)

    #Expand failedUrl
    session.failedUrl.extend(newlyFailedUrl)


#--Identify failed and corrupted items from img result
#Only use it when the session failed
def download_identifyFailure8Corruption(data, session, lastTargetId=11857, urlIdRange=False):
    session.corruptedUrl = ImgDownloader.Downloader.identifyCorruptions() #Examine the HHD as external is too slow
    session.currentDownloadId, session.failedUrl = ImgDownloader.Downloader.identifyFailures(data.urlInfo, lastTargetId, urlIdRange) #Examine the external


#--Download failed and corrupted image
def download_reDownload(data, session):
    #Perform redownload and update the failed list
    session.corruptedUrl = ImgDownloader.Downloader.download8SaveFailed(data.urlInfo, session.corruptedUrl)
    session.failedUrl = ImgDownloader.Downloader.download8SaveFailed(data.urlInfo, session.failedUrl)