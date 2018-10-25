import requests
import yaml
import pandas as pd
import pickle
from pprint import pprint
from bidict import bidict
import bin.module.util as util


#--Initialize data container
data = util.UniversalContainer()


#--Prepare name-id 2-way mapping
class Mapping():

    def generate(): #Process from source data
        df = pd.read_csv('data/text/df_cb_main_combined.csv', usecols=['Game']).drop_duplicates().reset_index(drop=True)
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
data.mapping = Mapping.generate()


#--Acquire img paths (Bing)
class ImgSearch():

    #Bing setup
    CRED = util.getConfigObj('ref/credential.yml')
    setting = util.SettingContainer(
        url=CRED.BingImageSearch.url,
        headers={'Ocp-Apim-Subscription-Key' : CRED.BingImageSearch.key},
        params={'q': '', 'license': 'public', 'imageType': 'photo'}
    )

    @classmethod
    def search(cls, targetTerm):
        cls.setting.params['q'] = targetTerm
        response = requests.get(cls.setting.url, headers=cls.setting.headers, params=cls.setting.params)
        response.raise_for_status()
        return response.json()
    
    @classmethod
    def test(cls):
        response = cls.search('puppies')
        pprint(response)


for i in mapping.inv: print(i)



#--Download img
img_url = 'http://images.fineartamerica.com/images-medium-large-5/abstract-art-original-painting-winter-cold-by-madart-megan-duncanson.jpg'
path = 'data/img/img_url1.jpeg'

r = requests.get(img_url, stream=True)
if r.status_code == 200:
    with open(path, 'wb') as f:
        for chunk in r.iter_content(1024): #'1024 = chunk size
            f.write(chunk)
