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
            session.currentId = 0
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
    def search(cls, targetTerm):
        cls.setting.params['q'] = targetTerm
        response = requests.get(cls.setting.url, headers=cls.setting.headers, params=cls.setting.params)
        response.raise_for_status()
        return response.json()
    
    @classmethod
    def searchMappingBatch(cls, mapping, startId, currentResponses=[]):
        ids = list(mapping)[startId:5]
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
    
    @classmethod
    def test(cls):
        #Check Append Results for Jupyter to render
        response = cls.search('puppies')
        pprint(response)
    
    def parseResponse(response):
        urls = 1
        return response['targetId'], urls
    
    @classmethod
    def parseResponses(cls, responses):
        urlInfo = []
        for response in responses:
            urlInfo.append(cls.parseResponse(response))
        return urlInfo

Store.data.responses, Store.session.currentId = ImgSearch.searchMappingBatch(Store.data.mapping, Store.session.currentId)
utils.writeJsls(Store.data.responses, 'data/image-response.jsl')

Store.data.responses = utils.readJsls('data/image-response.jsl')
len(Store.data.responses)
test[4]

Store.data.urlInfo = ImgSearch.parseResponses(Store.data.responses)


#--Download img
class ImgDownload():

    img_url = 'http://images.fineartamerica.com/images-medium-large-5/abstract-art-original-painting-winter-cold-by-madart-megan-duncanson.jpg'
    path = 'data/img/img_url1.jpeg'

    r = requests.get(img_url, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in r.iter_content(1024): #'1024 = chunk size
                f.write(chunk)
