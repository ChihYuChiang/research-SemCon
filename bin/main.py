import pickle
import bin.module.ImgDownloader as ImgDownloader
import bin.module.ImgPreprocessor as ImgPreprocessor
import bin.module.util as util
from bin.setting import path


#--Initialize session and data storage
data = util.UniversalContainer()
session = util.Session.load(path.session,
    currentDownloadId=0,
    currentSearchId=0
)


#--Create mapping, search, and download image
def imgDownload():

    #Create id and game title mapping
    data.mapping = ImgDownloader.Mapping.generate()


    #--Search image
    if False:
        #Perform search
        data.responses, session.currentSearchId = ImgDownloader.Searcher.searchBatch(data.mapping, startId=session.currentSearchId, batchSize=400)

        #Save search responses to file
        util.writeJsls(data.responses, path.imageResponse)


    #--Parse response
    if False:
        #Load search responses from file
        data.responses = util.readJsls(path.imageResponse)

        #Parse responses for url info
        data.urlInfo = ImgDownloader.Searcher.parseResponse_n(data.responses)

        #Save url info to file
        with open(path.imageUrl, 'wb') as f: pickle.dump(data.urlInfo, f)


    #--Download image
    #Load url info from file
    with open(path.imageUrl, 'rb') as f: data.urlInfo = pickle.load(f)

    #Perform download
    session.currentDownloadId, session.failedUrl = ImgDownloader.Downloader.download8SaveBatch(data.urlInfo, startId=session.currentDownloadId, batchSize=3, urlIdRange=[95, 100])


    #--End session
    #Store session info offline
    session.dump(path.session)


def textPreprocess():
    #sentence vec

    #tokenized main, with exp only, remove stop, remove nonword

    #tokenized comment, with exp + stop, keep nonword

    #Tokenized comment, with exp only

    #List of exp word

    #emb - word 2 way table


#--Process image into cleaned format for input
def imgPreprocess():
    #Cropping to be square

    #Scaling to 100px by 100px

    #Img selection: separate gameplay, logo?

    #Mean, standard deviation of input pixel

    #Normalizing

    #Augmentation: Perturbation, rotation