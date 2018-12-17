import requests
import pandas as pd
import pickle
import traceback
import re
from pprint import pprint
from bidict import bidict
from os import listdir

import bin.module.util as util
from bin.setting import path, imgDownloader as config


#--Prepare name-id 2-way mapping
class Mapping():

    #Process from source data
    def generate():
        df = pd.read_csv(path.textDfCombined, usecols=['Game'], keep_default_na=False).drop_duplicates().reset_index(drop=True)
        return bidict(df.to_dict()['Game'])

    @classmethod
    def export(cls):
        with open(path.mapping, 'wb') as f: pickle.dump(cls.generate(), f)
    
    @classmethod
    def test_generate(cls):
        mapping = cls.generate()
        print(mapping[1])
        print(mapping.inv['Expeditions: Viking'])
        print(len(mapping))


#--Acquire img urls (Bing)
class Searcher():

    #Bing setup within cls to improve performance
    #https://docs.microsoft.com/en-us/rest/api/cognitiveservices/bing-images-api-v7-reference
    logger = util.initLogger(loggerName='ImgDownloader.Searcher')
    params = config.searcherParams.copy()
    headers = config.searcherHeaders
    searchUrl = config.searcherUrl

    @classmethod
    @util.FuncDecorator.delayOperation(1)
    def search(cls, targetTerm):
        cls.params['q'] = targetTerm
        response = requests.get(cls.searchUrl, headers=cls.headers, params=cls.params)

        response.raise_for_status()
        cls.logger.debug('Searched \"{}\".'.format(targetTerm))
        return response.json()

    @classmethod
    def searchBatch(cls, mapping, startId, batchSize):
        ids = list(mapping)[startId : startId + batchSize] #Python tolerates slicing index go over len
        idIter = iter(ids)
        responses = []
    
        while True:
            try:
                targetId = next(idIter)
                targetTerm = '{} game'.format(mapping[targetId]) #'Add 'game' at the end of each title 
                response = cls.search(targetTerm)
                response['targetId'] = targetId
                responses.append(response)
            except StopIteration:
                cls.logger.info('Finished searches at {} (included).'.format(targetId))
                return responses, targetId + 1
            except:
                cls.logger.error('Unexpected error - {}:\n{}'.format(targetTerm, traceback.format_exc()))
                return responses, targetId
    
    def parseResponse_1(response):
        urls = []
        for i in range(len(response['value'])):
            urls.append((i, response['value'][i]['contentUrl']))
        return response['targetId'], urls
    
    @classmethod
    def parseResponse_n(cls, responses):
        urlInfo = []
        for response in responses:
            urlInfo.append(cls.parseResponse_1(response))
        cls.logger.info('Parsed {} response items.'.format(len(urlInfo)))
        return urlInfo
    
    @classmethod
    def test_search(cls):
        #Check Append Results for Jupyter to render
        response = cls.search('puppies')
        pprint(response)


#--Download img
class Downloader():

    #Init logger
    logger = util.initLogger(loggerName='ImgDownloader.Downloader')
    imageFolder = path.imageFolder

    def retrieveUrlEntry(urlInfo, targetId):
        #Find the entry of the target Id
        return next((entry for entry in urlInfo if entry[0] == targetId), None) #If not found, return None
    
    @classmethod    
    def identifyFailures(cls, lastTargetId, urlIdRange):
        #TODO: identify bad img file (can't read)
        #E.g. 48-56.jpg in folder, lastTargetId=48
        urlIdRange = urlIdRange or [0, 100] 

        fullImgs = []
        for i in range(lastTargetId + 1):
            for j in range(*urlIdRange):
                fullImgs.append((i, j))

        curImgs = []
        curImgNames = listdir(path.imageFolder)
        for name in curImgNames:
            match = re.match('(\d+)-(\d+)\.', name)
            curImgs.append((int(match.group(1)), int(match.group(2))))
        
        failedUrl = list(set(fullImgs) - set(curImgs))

        cls.logger.info('Identified {} failed downloads.'.format(len(failedUrl)))
        return lastTargetId + 1, failedUrl

    @classmethod
    @util.FuncDecorator.delayOperation(1)
    def download(cls, targetId, url):
        try:
            response = requests.get(url, stream=True, timeout=5, headers=util.createCustomHeader()) #Time out to stop waiting for a response
            response.raise_for_status()
            return response
        except:
            cls.logger.error('Unexpected error - {} at {}:\n{}'.format(targetId, url, traceback.format_exc()))
            return False

    @classmethod
    def save(cls, response, targetId, urlId):
        try:
            fileName = '{}-{}.jpg'.format(targetId, urlId)
            path = '{}{}'.format(cls.imageFolder, fileName)
            with open(path, 'wb') as f:
                for chunk in response.iter_content(1024): #'1024 = chunk size
                    f.write(chunk)
            cls.logger.debug('Downloaded {}.'.format(fileName))
            return True
        except:
            cls.logger.error('Unexpected error - {} when saving {}:\n{}'.format(targetId, urlId, traceback.format_exc()))
            return False


    @classmethod
    def download8SaveBatch(cls, urlInfo, startId, batchSize, urlIdRange):
        #urlIdRange = [0, 10] or False for all ids

        failedItems = []
        #Download certain range of urlId of a target
        for targetId in range(startId, startId + batchSize):
            targetEntry = cls.retrieveUrlEntry(urlInfo, targetId)

            #Make sure the id is found
            try: assert targetEntry
            except AssertionError:
                errMsg = 'TargetId {} did not found in `urlInfo`.'.format(targetId[startId][0], startId)
                cls.logger.error('Assertion error - ' + errMsg)
                raise AssertionError(errMsg)

            for urlId, url in (targetEntry[1][urlIdRange[0]:urlIdRange[1]] if urlIdRange else targetEntry[1]):

                response = cls.download(targetId, url)
                if not response: failedItems.append((targetId, urlId)); continue

                saved = cls.save(response, targetId, urlId)
                if not saved: failedItems.append((targetId, urlId))
        
        cls.logger.info('Finished downloads at {} (included).\nAccumulated {} failed items.'.format(targetId, len(failedItems)))
        return targetId + 1, failedItems
    
    @classmethod
    def download8SaveFailed(cls, urlInfo, failedItems):
        failedItems_updated = []
        for failedItem in failedItems:
            targetEntry = cls.retrieveUrlEntry(urlInfo, failedItem[0])
            url = targetEntry[1][failedItem[1]][1]

            response = cls.download(failedItem[0], url)
            if not response: failedItems_updated.append(failedItem); continue

            saved = cls.save(response, *failedItem)
            if not saved: failedItems_updated.append(failedItem)
        return failedItems_updated
    
    @classmethod
    def test_download(cls):
        from PIL import Image
        from io import BytesIO

        response = cls.download('t', 'https://assets-cdn.github.com/images/modules/open_graph/github-mark.png')
        img = Image.open(BytesIO(response.content))
        img.show()