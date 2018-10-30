import nltk
import bin.module.util as util

cred = util.getConfigObj('ref/credential.yml')
path = util.SettingContainer(
    session        = 'data/session.pkl',
    mapping        = 'data/name-id-mapping.pkl',
    textDf         = 'data/text/df_cb_main.csv', #Include only GameSpot
    textDfCombined = 'data/text/df_cb_main_combined.csv', #Include all 3 sites
    imageResponse  = 'data/image-response.jsl',
    imageUrl       = 'data/image-url.pkl',
    imageFolder    = 'data/img/'
)
config = util.SettingContainer(
    ImgDownloader=util.SettingContainer(
        searcherUrl=cred.BingImageSearch.url,
        searcherHeaders={
            'Ocp-Apim-Subscription-Key': cred.BingImageSearch.key
        },
        searcherParams={
            'q'          : '',
            'license'    : 'all',
            'imageType'  : 'photo',
            'count'      : 100,
            'safeSearch' : 'off',
            'maxFileSize': 520192,
            'minFileSize': 0  #byte
        },
        downloaderHeaders={'user-agent': 'my-app/0.0.1'}
    ),

    TextPreprocessor=util.SettingContainer(
        stopWords=nltk.corpus.stopwords.words('english'),
        stemmer=nltk.stem.snowball.SnowballStemmer('english'),
        lemmatizer=nltk.stem.WordNetLemmatizer(),
        words2Filter=[],
        words2Keep=[]
    )
)