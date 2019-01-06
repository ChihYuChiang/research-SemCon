'''
Train a recurrent convolutional network on the IMDB sentiment
classification task.
Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.

Note:
batchSize is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''
import bin.module.TextPreprocessor as TextPreprocessor
import bin.module.util as util
from bin.setting import path

# from keras.preprocessing import sequence
# from keras.models import Model
# from keras.layers import Input, Dense, Dropout
# from keras.layers import Embedding, LSTM
# from keras.layers import Conv1D, MaxPooling1D

# emb, _ = TextPreprocessor.EmbOperator.loadPretrainedEmb8Keywords(path.gNewsW2V)
dfDispatcher = TextPreprocessor.DfDispatcher(path.textIMDBDf)

class IMDBReader():
    @staticmethod
    def readAsDf():
        import pandas as pd
        import os
        import re

        dic = {}
        urlFiles = {}
        for (dirpath, dirnames, filenames) in os.walk(path.textIMDBFolder, topdown=True):
            group_orig = 'train' if re.search('train', dirpath) else 'test' #The original train/test seperation by the author
            positive = not re.search('neg', dirpath)
            urlFile = '{}-{}'.format(group_orig, str(positive))
            
            for filename in filenames:
                match = re.match('^(\d+)_(\d+)\.txt$', filename)
                filepath = os.path.join(dirpath, filename)
                
                #If the particular filename format, read txt file into df
                if match:
                    title = re.search('title/(.+)/', urlFiles[urlFile][int(match.group(1))]).group(1)
                    rating = match.group(2)
                    with open(filepath, 'rt') as f:
                        text = f.read()
                    dic['{}'.format(re.search('(.+)/(.+)\.', filepath).group(2))] = [title, rating, text, positive, group_orig]
                
                #Get title id from the url files
                elif re.match('urls', filename):
                    with open(filepath, 'rt') as f:
                        positive_url = str(not re.search('neg', filename))
                        urlFiles['{}-{}'.format(group_orig, positive_url)] = list(f) #Turn into a list, each line an element
                        
        df = pd.DataFrame.from_dict(dic, orient='index')
        df.columns = ['title', 'rating', 'text', 'positive', 'group_orig']

        print('Read in {} reviews.'.format(df.shape[0]))
        return df
    
    @classmethod
    def exportDf(cls):
        cls.readAsDf().to_csv(path.textIMDBDf)

# IMDBReader.exportDf()

TextPreprocessor.Tokenizer.brief(dfDispatcher.getCol('text'))
articles_tokenized = TextPreprocessor.Tokenizer.tokenize(dfDispatcher.getCol('text'))

util.createListFromGen(articles_tokenized)


normalizer = TextPreprocessor.Normalizer(articles_tokenized)
articles_normalized = normalizer.lower().filterStop().filterNonWord().getResult()



#--Configuration
#Data
vocabSize = 0
maxlen = 200 #Use only the first 200 words

#Model
dropoutRate = 0.25
poolSize = 4
config_conv1D = {'filters': 64, 'kernel_size': 5}
config_LSTM = {'units': 64}

#Training
batchSize = 32
epochs = 1


#--Data
id_train, id_test = util.divideSets([], nSample)
#load the particular file
#text preprocessor

#Pad sequence
config_padSequence = {'maxlen': maxlen, padding: 'post', truncating: 'post'}
x_train = sequence.pad_sequences(x_train, **config_padSequence)
x_test = sequence.pad_sequences(x_test, **config_padSequence)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

embWeights


#--Model
EmbWPresetWeight = Embedding(input_dim=vocabSize, output_dim=300)
EmbWPresetWeight.set_weights(embWeights)

inputs = Input(shape=(maxlen,), dtype='int32')
_ = EmbWPresetWeight(inputs)
_ = Dropout(dropoutRate)(_)
_ = Conv1D(**config_conv1D, strides=1, padding='valid', activation='relu')(_)
_ = MaxPooling1D(poolSize)(_)
_ = LSTM(**config_LSTM)(_)
outputs = Dense(1, activation='linear')(_)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='logcosh', optimizer='adam', metrics=['logcosh']) #TODO: customize the metric


#--Training
model.fit(x_train, y_train, batchSize=batchSize, epochs=epochs)


#--Evaluation
score, acc = model.evaluate(x_test, y_test, batchSize=batchSize)
print('Test score:', score)
print('Test accuracy:', acc)