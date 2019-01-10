import nltk.corpus
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

import bin.module.util as util


cred = util.general.getConfigObj('ref/credential.yml')
path_dataLake = 'E:/Repository/Data/' #External data storage
path = util.general.SettingContainer(
    session        = 'data/session.pkl',
    mapping        = 'data/name-id-mapping.pkl',
    textFolder     = 'data/text/',
    textTkFolder   = 'data/text/tokenized/',
    textDictFolder = 'data/text/dictionary/',
    textDf         = 'data/text/df_cb_main.csv',          #Include only GameSpot
    textDfCombined = 'data/text/df_cb_main_combined.csv', #Include all  3 sites
    textIMDBFolder = 'data/text/aclImdb/',
    textIMDBDf     = 'data/text/df_imdb.csv',
    gNewsW2V       = 'data/text/emb/GoogleNews-vectors-negative300.bin.gz',
    expKeyword     = 'data/text/keywords.csv',
    imageUrl       = 'data/image-url.pkl',
    imageResFolder = 'data/img-response/',
    imageFolder    = 'data/img/',
    modelFolder    = 'model/',
    
    dataLake = util.general.SettingContainer(
        imageResFolder = path_dataLake + 'Game Image/img-response/',
        imageFolder = path_dataLake + 'Game Image/img/'
    )
)

imgDownloader = util.general.SettingContainer(
    searcherUrl = cred.BingImageSearch.url,
    searcherHeaders = {
        'Ocp-Apim-Subscription-Key': cred.BingImageSearch.key
    },
    searcherParams = {
        'q'          : '',
        'license'    : 'all',
        'imageType'  : 'photo',
        'count'      : 100,
        'safeSearch' : 'off',
        'maxFileSize': 520192,
        'minFileSize': 0  #byte
    }
)

textPreprocessor = util.general.SettingContainer(
    stopwords    = nltk.corpus.stopwords.words('english'),
    stemmer      = SnowballStemmer('english').stem,
    lemmatizer   = WordNetLemmatizer().lemmatize,
    words2Filter = ['br'],
    words2Keep   = []
)

textSummarizer = util.general.SettingContainer(
    modelSentimentParams = util.general.SettingContainer(
        #Data
        vocabSize = 0,
        config_padSequence = {
            'maxlen': 200, #Use only the first 200 words
            'padding': 'post',
            'truncating': 'post'
        },

        #Model
        embWeightInit = '',
        dropoutRate = 0.25,
        poolSize = 4,
        config_conv1D = {
            'filters': 64,
            'kernel_size': 5
        },
        config_LSTM = {
            'units': 64
        },

        #Compile
        loss='logcosh', optimizer='adam', metrics=['logcosh'],

        #Training
        batchSize = 32, epochs = 1
    )
)
