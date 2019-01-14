import pickle

import bin.module.util as util
from bin.setting import path

from . import img, text


logger = util.general.initLogger(loggerName='pipeline')


#--Initialization
#Provide the attributes to be overwritten in an obj
def initialize(sessionPath, overwrite=False):
    data = util.general.UniversalContainer()
    model = util.general.UniversalContainer()

    try:
        #Load id and game title mapping
        with open(path.mapping, 'rb') as f: data.mapping = pickle.load(f)

        #Load url info from file
        with open(path.imageUrl, 'rb') as f: data.urlInfo = pickle.load(f)
    except OSError: logger.warning('Either `mapping` or `urlInfo` file not found. This can cause error in later operations.')
    
    #Load session. If no file found, create new session with the template
    newSession = {
        'currentDownloadId': 0,
        'currentSearchId': 0,
        'modelSentimentEpoch': 0
    }
    session = util.general.Session.load(sessionPath, **newSession)
    
    return data, model, session


#--Observe session outcome
def observeOutcome(data, model, session):
    logger.info('-' * 60)
    logger.info('session')
    logger.info(session)
    logger.info('-' * 60)
    logger.info('data')
    logger.info(data)
    logger.info('-' * 60)
    logger.info('model')
    logger.info(model)
    logger.info('-' * 60)