import bin.module.pipeline as pipeline
from bin.setting import path


#--Initialize session and data storage
data, model, session = pipeline.initialize(path.session)


#--Operations
'''
pipeline.img.download_search(data, session, batchSize=2)
pipeline.img.download_parse(data)
pipeline.img.download_download(data, session, batchSize=2)
pipeline.img.download_identifyFailure8Corruption(data, session)
pipeline.img.download_reDownload(data, session)

pipeline.text.preprocess_initSentiment(data, load=True)
pipeline.text.summarize_initSentiment(data, model, session, load=True)
pipeline.text.summarize_trainSentiment(data, model, session, epochs=3)
pipeline.text.summarize_predictSentiment([''], model)
'''


#--Observe session outcome
pipeline.observeOutcome(data, model, session)


#--End session
#TODO: Slack notification
#https://medium.com/@koitaroh/make-notifications-with-slack-api-when-python-experiment-is-done-c74539c1e4e9
#Store session info offline
session.dump(path.session)