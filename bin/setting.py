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
    textEmbFolder  = 'data/text/emb/',
    textDf         = 'data/text/df_cb_main.csv', #Only GameSpot, encoding cp1252
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
        batchSize = 32,
        config_padSequence = {
            'maxlen': 200, #Use only the first 200 words
            'padding': 'post',
            'truncating': 'post'
        },

        #Model
        embSize = 300,
        embTranable = True,
        dropoutRate = 0.25,
        poolSize = 4,
        config_conv1D = {
            'filters': 64,
            'kernel_size': 5
        },
        LSTMUnits = 64,

        #Compile
        config_compile = {
            'loss': 'logcosh',
            'optimizer': 'adam',
            'metrics': ['mean_absolute_error']
        },

        #Training
        config_training = {
            'epochs': 1,
            'shuffle': True #After each epoch
        }
    ),
    modelEncoderDecoderParams = util.general.SettingContainer(
        #Multiprocessing
        config_multiprocessing = {
            'use_multiprocessing': True,
            'workers': 6
        },

        #Model
        encoderEmb = {
            'size': 300,
            'trainable': False
        },
        dropoutRate = 0.25,
        poolSize = 4,
        config_conv1D = {
            'filters': 64,
            'kernel_size': 5
        },
        LSTMUnits = 256,
        decoderEmb = {
            'size': 300,
            'trainable': True
        },

        #Compile
        config_compile = {
            'loss': 'categorical_crossentropy',
            'optimizer': 'adam'
        },

        #Training
        config_training = {
            'epochs': 1
        }
    )
)
