import bin.module.pipeline as pipeline
from bin.setting import path


#--Initialize session and data storage
data, model, session = pipeline.initialize(path.session)


#--Operations
'''
pipeline.imgDownload_search(data, session, batchSize=2)
pipeline.imgDownload_parse(data)
pipeline.imgDownload_download(data, session, batchSize=2)
pipeline.imgDownload_identifyFailure8Corruption(data, session)
pipeline.imgDownload_reDownload(data, session)

pipeline.textPreprocess_initSentiment(data)
pipeline.textSummarize_initSentiment(data, model)
pipeline.textSummarize_trainSentiment(data, model, session, epochs=1)
pipeline.textSummarize_predictSentiment(text, model)
'''


#--Observe session outcome
pipeline.observeOutcome(data, model, session)


#--End session
#TODO: Slack notification
#https://medium.com/@koitaroh/make-notifications-with-slack-api-when-python-experiment-is-done-c74539c1e4e9
#Store session info offline
session.dump(path.session)