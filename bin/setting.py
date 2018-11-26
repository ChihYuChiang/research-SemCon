import nltk
import bin.module.util as util


cred = util.getConfigObj('ref/credential.yml')
path = util.SettingContainer(
    session        = 'data/session.pkl',
    mapping        = 'data/name-id-mapping.pkl',
    textDf         = 'data/text/df_cb_main.csv', #Include only GameSpot
    textDfCombined = 'data/text/df_cb_main_combined.csv', #Include all 3 sites
    gNewsW2V       = 'data/text/GoogleNews-vectors-negative300.bin.gz',
    expKeyword     = 'data/text/keywords.csv',
    imageResponse  = 'data/image-response.jsl',
    imageUrl       = 'data/image-url.pkl',
    imageFolder    = 'data/img/'
)

imgDownloader = util.SettingContainer(
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
)

textPreprocessor = util.SettingContainer(
    stopwords=nltk.corpus.stopwords.words('english'),
    stemmer=nltk.stem.snowball.SnowballStemmer('english').stem,
    lemmatizer=nltk.stem.WordNetLemmatizer().lemmatize,
    words2Filter=[],
    words2Keep=[]
)
