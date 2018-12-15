import bin.module.pipeline as pipeline
from bin.setting import path


#--Initialize session and data storage
data, session = pipeline.initialize(path.session) 


#--Operations
'''
pipeline.imgDownload_search(data, session, batchSize=10)
pipeline.imgDownload_parse(data)
pipeline.imgDownload_download(data, session, batchSize=3)
'''
pipeline.imgDownload_download(data, session, batchSize=3)


#--Observe session outcome
pipeline.observeOutcome(data, session)


#--End session
#Store session info offline
session.dump(path.session)