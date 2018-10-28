import requests
import pandas as pd
import pickle
import sys
from pprint import pprint
from bidict import bidict
import bin.module.util as util


#--Initialize data storage
#Also create `session` attribute in the class
class Store():

    #Initialize store content
    def initSession(path):
        try:
            with open(path, 'rb') as f:
                session = pickle.load(f, encoding='utf-8')
                print('Loaded session at \'{}\'.\n{}'.format(path, session))
        except:
            session = util.UniversalContainer()
            session.currentSearchId = 0
            session.currentDownloadId = 0
            print('Did not find session at \'{}\'. Initiated a new session.\n{}'.format(path, session))
        return session

    config = util.SettingContainer(
        cred=util.getConfigObj('ref/credential.yml'),
        path=util.SettingContainer()
    )
    config.path.update(
        session='data/session.pkl',
        imageResponse='data/image-response.jsl',
        imageUrl='data/image-url.pkl'
    )
    data = util.UniversalContainer()
    session = initSession(config.path.session)

    @classmethod
    def dumpSession(cls, path):
        with open(path, 'wb') as f: pickle.dump(cls.session, f)
        print('Dumped session at \'{}\'.\n{}'.format(path, cls.session))

    #Show all store content
    @classmethod
    def reveal(cls):
        exp = '<session>\n{}\n\n<config>\n{}\n\n<data>\n{}'.format(cls.session, cls.config, cls.data)
        print(exp)


#--Prepare name-id 2-way mapping
class Mapping():

    #Process from source data
    def generate():
        df = pd.read_csv('data/text/df_cb_main_combined.csv', usecols=['Game'], keep_default_na=False).drop_duplicates().reset_index(drop=True)
        return bidict(df.to_dict()['Game'])

    @classmethod
    def export(cls):
        with open('data/name-id-mapping.pkl', 'wb') as f: pickle.dump(cls.generate(), f)
    
    @classmethod
    def test(cls):
        mapping = cls.generate()
        print(mapping[1])
        print(mapping.inv['Expeditions: Viking'])
        print(len(mapping))


#--Acquire img urls (Bing)
class ImgSearch():

    #Bing setup and init logger
    #https://docs.microsoft.com/en-us/rest/api/cognitiveservices/bing-images-api-v7-reference
    logger = util.initLogger(loggerName='ImgSearch')
    setting = util.SettingContainer(
        url=Store.config.cred.BingImageSearch.url,
        headers={
            'Ocp-Apim-Subscription-Key': Store.config.cred.BingImageSearch.key
        },
        params={
            'q': '', 'license': 'all', 'imageType': 'photo',
            'count': 100, 'safeSearch': 'off',
            'maxFileSize': 520192, 'minFileSize':0 #byte
        }
    )

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
                targetTerm = '{} game'.format(Store.data.mapping[targetId]) #'Add 'game' at the end of each title 
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
class ImgDownload():

    #Init logger and http request header
    logger = util.initLogger(loggerName='ImgDownload')
    headers = {'user-agent': 'my-app/0.0.1'}

    @classmethod
    @util.FuncDecorator.delayOperation(1)
    def get8Save_1(cls, targetId, urlId, url):
        try:
            r = requests.get(url, stream=True, timeout=5) #Time out to stop waiting for a response
            r.raise_for_status()
        except:
            cls.logger.error("Unexpected error - {} at {}:\n{}".format(targetId, url, sys.exc_info()[0]))
            return False
            
        fileName = '{}-{}.jpg'.format(targetId, urlId)
        path = 'data/img/{}'.format(fileName)
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


#--Implementation
def main():

    #Create id and game title mapping
    Store.data.mapping = Mapping.generate()


    #--Search image
    if False:
        #Perform search
        Store.data.responses, Store.session.currentSearchId = ImgSearch.searchMappingBatch(Store.data.mapping, startId=Store.session.currentSearchId, batchSize=1000)

        #Save search responses to file
        util.writeJsls(Store.data.responses, Store.config.path.imageResponse)


    #--Parse response
    if False:
        #Load search responses from file
        Store.data.responses = util.readJsls(Store.config.path.imageResponse)

        #Parse responses for url info
        Store.data.urlInfo = ImgSearch.parseResponse_n(Store.data.responses)

        #Save url info to file
        with open(Store.config.path.imageUrl, 'wb') as f: pickle.dump(Store.data.urlInfo, f)


    #--Download image
    #Load url info from file
    with open(Store.config.path.imageUrl, 'rb') as f: Store.data.urlInfo = pickle.load(f)

    #Perform download
    Store.session.currentDownloadId, Store.session.failedUrl = ImgDownload.get8Save_n(Store.data.urlInfo, startId=Store.session.currentDownloadId, batchSize=3, urlIdRange=[95, 100])


    #--End session
    #Store session info offline
    Store.dumpSession(Store.config.path.session)