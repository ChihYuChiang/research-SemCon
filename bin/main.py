import bin.module.pipeline as pipeline
from bin.setting import path


#--Initialize session and data storage
data, session = pipeline.initialize(path.session)


#--Operations
'''
pipeline.imgDownload_search(data, session, batchSize=2)
pipeline.imgDownload_parse(data)
pipeline.imgDownload_download(data, session, batchSize=2)
pipeline.imgDownload_identifyFailures(data, session)
pipeline.imgDownload_identifyCorrupted(session)
pipeline.imgDownload_reDownload(data, session)
'''


#--Observe session outcome
pipeline.observeOutcome(data, session)


#--End session
#TODO: Slack notification
#https://medium.com/@koitaroh/make-notifications-with-slack-api-when-python-experiment-is-done-c74539c1e4e9
#Store session info offline
session.dump(path.session)