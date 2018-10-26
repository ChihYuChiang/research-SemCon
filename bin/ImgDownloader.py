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
            with open(path, 'rb') as f: return pickle.load(f, encoding='utf-8')
        except:
            session = util.UniversalContainer()
            session.currentSearchId = 0
            session.currentDownloadId = 0
            return session

    config = util.SettingContainer(
        cred=util.getConfigObj('ref/credential.yml'),
        sessionPath='data/session.pkl'
    )
    data = util.UniversalContainer()
    session = initSession(config.sessionPath)

    @classmethod
    def dumpSession(cls, path):
        with open(path, 'wb') as f: pickle.dump(cls.session, f)
    
    @classmethod
    def loadSession(cls, path):
        with open(path, 'rb') as f: cls.session = pickle.load(f, encoding='utf-8')    

    #Show all store content
    @classmethod
    def reveal(cls):
        exp = '<session>\n{}\n\n<config>\n{}\n\n<data>\n{}'.format(cls.session, cls.config, cls.data)
        print(exp)

#Store session info offline
Store.dumpSession(Store.config.sessionPath)


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

#Mapping data
Store.data.mapping = Mapping.generate()


#--Acquire img urls (Bing)
class ImgSearch():

    #Bing setup
    #https://docs.microsoft.com/en-us/rest/api/cognitiveservices/bing-images-api-v7-reference
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
        print('Searched \"{}\".'.format(targetTerm))
        return response.json()
    
    @classmethod
    def test_search(cls):
        #Check Append Results for Jupyter to render
        response = cls.search('puppies')
        pprint(response)

    @classmethod
    def searchMappingBatch(cls, mapping, startId, batchSize=5, currentResponses=[]):
        ids = list(mapping)[startId : max([startId + batchSize, len(mapping)])]
        idIter = iter(ids)
        responses = currentResponses
    
        while True:
            try:
                targetId = next(idIter)
                targetTerm = '{} game'.format(Store.data.mapping[targetId]) 
                response = cls.search(targetTerm)
                response['targetId'] = targetId
                responses.append(response)
            except StopIteration: return responses, targetId + 1
            except:
                print("Unexpected error:", sys.exc_info()[0])
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

# Store.data.responses, Store.session.currentSearchId = ImgSearch.searchMappingBatch(Store.data.mapping, Store.session.currentSearchId)
# util.writeJsls(Store.data.responses, 'data/image-response.jsl')

Store.data.responses = util.readJsls('data/image-response.jsl')
Store.data.urlInfo = ImgSearch.parseResponse_n(Store.data.responses)

@util.FuncDecorator.delayOperation(3)
def get8Save_1(targetId, imgId, url):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        fileName = '{}-{}.jpg'.format(targetId, imgId)
        path = 'data/img/{}'.format(fileName)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(1024): #'1024 = chunk size
                f.write(chunk)
        print('Downloaded {}'.format(fileName))
        return 
    else: pass
        #Save failed ids and urls in session

def get8Save_n(urlInfo):
    for item in urlInfo:
        targetId = item[0]
        for url8Id in item[1][:5]: get8Save_1(targetId, *url8Id)

Store.session.currentSearchId = get8Save_n(Store.data.urlInfo)

#--Download img
class ImgDownload():