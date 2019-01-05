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

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, MaxPooling1D


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
outputs = Dense(1, activation='sigmoid')(_)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#--Training
model.fit(x_train, y_train, batchSize=batchSize, epochs=epochs)


#--Evaluation
score, acc = model.evaluate(x_test, y_test, batchSize=batchSize)
print('Test score:', score)
print('Test accuracy:', acc)