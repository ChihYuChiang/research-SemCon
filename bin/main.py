import pickle
import bin.module.ImgDownloader as ImgDownloader
import bin.module.ImgPreprocessor as ImgPreprocessor
import bin.module.util as util
import bin.setting as setting


#--Initialize session and data storage
data = util.UniversalContainer()
session = util.Session.load(setting.path.session,
    currentDownloadId=0,
    currentSearchId=0
)


#--Create mapping, search, and download image
def imgDownload():

    #Create id and game title mapping
    data.mapping = Mapping.generate()


    #--Search image
    if False:
        #Perform search
        data.responses, session.currentSearchId = ImgSearch.searchBatch(data.mapping, startId=session.currentSearchId, batchSize=400)

        #Save search responses to file
        util.writeJsls(data.responses, setting.path.imageResponse)


    #--Parse response
    if False:
        #Load search responses from file
        data.responses = util.readJsls(setting.path.imageResponse)

        #Parse responses for url info
        data.urlInfo = ImgSearch.parseResponse_n(data.responses)

        #Save url info to file
        with open(setting.path.imageUrl, 'wb') as f: pickle.dump(data.urlInfo, f)


    #--Download image
    #Load url info from file
    with open(setting.path.imageUrl, 'rb') as f: data.urlInfo = pickle.load(f)

    #Perform download
    session.currentDownloadId, session.failedUrl = ImgDownload.get8SaveBatch(data.urlInfo, startId=session.currentDownloadId, batchSize=3, urlIdRange=[95, 100])


    #--End session
    #Store session info offline
    session.dump(setting.path.session)


#--Process image into cleaned format for input
def imgPreprocess():
    #Cropping to be square

    #Scaling to 100px by 100px

    #Img selection: separate gameplay, logo?

    #Mean, standard deviation of input pixel

    #Normalizing

    #Augmentation: Perturbation, rotation