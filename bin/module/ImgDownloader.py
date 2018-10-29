import requests
import pandas as pd
import pickle
import sys
from pprint import pprint
from bidict import bidict
import bin.module.util as util
import bin.setting as setting


#--Prepare name-id 2-way mapping
class Mapping():

    #Process from source data
    def generate():
        df = pd.read_csv(setting.path.textDf, usecols=['Game'], keep_default_na=False).drop_duplicates().reset_index(drop=True)
        return bidict(df.to_dict()['Game'])

    @classmethod
    def export(cls):
        with open(setting.path.mapping, 'wb') as f: pickle.dump(cls.generate(), f)
    
    @classmethod
    def test(cls):
        mapping = cls.generate()
        print(mapping[1])
        print(mapping.inv['Expeditions: Viking'])
        print(len(mapping))


#--Acquire img urls (Bing)
class Searcher():

    #Bing setup and init logger
    #https://docs.microsoft.com/en-us/rest/api/cognitiveservices/bing-images-api-v7-reference
    logger = util.initLogger(loggerName='ImgDownloader.Searcher')

    @classmethod
    @util.FuncDecorator.delayOperation(1)
    def search(cls, targetTerm):
        cls.setting.params['q'] = targetTerm
        response = requests.get(cls.setting.url, headers=cls.setting.headers, params=cls.setting.params)

        response.raise_for_status()
        cls.logger.debug('Searched \"{}\".'.format(targetTerm))
        return response.json()
    
    @classmethod
    def test_search(cls):
        #Check Append Results for Jupyter to render
        response = cls.search('puppies')
        pprint(response)

    @classmethod
    def searchMappingBatch(cls, mapping, startId, batchSize=5, currentResponses=[]):
        ids = list(mapping)[startId : startId + batchSize] #Python tolerates slicing index go over len
        idIter = iter(ids)
        responses = currentResponses
    
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
                cls.logger.error('Unexpected error - {}:\n{}'.format(targetTerm, sys.exc_info()[0]))
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
        return urlInfo


#--Download img
class Downloader():

    #Init logger and http request header
    logger = util.initLogger(loggerName='ImgDownloader.Downloader')

    @classmethod
    @util.FuncDecorator.delayOperation(1)
    def get8Save_1(cls, targetId, urlId, url):
        try:
            r = requests.get(url, stream=True, timeout=5) #Time out to stop waiting for a response
            r.raise_for_status()
        except:
            cls.logger.error('Unexpected error - {} at {}:\n{}'.format(targetId, url, sys.exc_info()[0]))
            return False
        
        #TODO: Make separate folders
        fileName = '{}-{}.jpg'.format(targetId, urlId)
        path = '{}{}'.format(setting.path.imageFolder, fileName)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(1024): #'1024 = chunk size
                f.write(chunk)
        cls.logger.debug('Downloaded {}.'.format(fileName))
        return True

    @classmethod
    def get8Save_n(cls, urlInfo, startId, batchSize=5, urlIdRange=[0, 10]):
        failedItems = []

        #Download certain range of urlId of a target
        for item in urlInfo[startId : startId + batchSize]:
            targetId = item[0]
            for url8Id in item[1][urlIdRange[0]:urlIdRange[1]]:
                success = cls.get8Save_1(targetId, *url8Id)
                if not success: failedItems.append((targetId, url8Id[0]))
        
        cls.logger.info('Finished downloads at {} (included).\nAccumulated {} failed items.'.format(targetId, len(failedItems)))
        return targetId + 1, failedItems