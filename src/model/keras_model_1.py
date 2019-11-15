#%% Run imports
import matplotlib.pyplot as plt
import seaborn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from nltk.corpus import stopwords
from keras.layers import Bidirectional
import matplotlib.pyplot as plt
import seaborn
# TODO:  Create a Model class

#%% Run Words
stop_words = set(stopwords.words('english'))
stop_words = {x.replace("'","") for x in stop_words if re.search("[']", x.lower())}

path = os.path.abspath(os.curdir)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
movie_data = pd.read_csv('{}/../csv-data/movie-data-cleaned.csv'.format(path))
print(len(movie_data))

movie_data = np.split(movie_data, [3], axis=1)
movie_data[0].drop(columns='Unnamed: 0', inplace=True)
features = movie_data[0]['Summary'].values.astype('U')


n_most_common_words = 10000
max_len = 500
tokenizer = Tokenizer(num_words=n_most_common_words, lower=True)
tokenizer.fit_on_texts(features)
sequences = tokenizer.texts_to_sequences(features)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
X = pad_sequences(sequences, maxlen=max_len, truncating="pre")

epochs = 10
emb_dim = 128
batch_size = 16

#%% Run Changes
genres = ['Action',
          'Comedy'#,
          # 'Adventure'
          # 'Drama',
          # 'Thriller',
          # 'Horror',
          # 'Romance',
          # 'Crime'
          ]
#%% Run Changes
def create_train_model(genre):
    y = movie_data[1][genre]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    model = Sequential()
    model.add(Embedding(n_most_common_words, emb_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.7))
    model.add(Bidirectional(LSTM(30, recurrent_dropout=0.7)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_test, y_test),callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])

    #for genre, history in hist.items():
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy {}'.format(genre))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss for {}'.format(genre))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()

    return model

action_model = create_train_model('Action')
comedy_model = create_train_model('Comedy')
#%% Run Test
y = movie_data[1]['Action']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
print(X_test)
#%% Run Test
print("Prediction")
action_pred = action_model.predict(X_test)

print(type(action_pred.argmax(axis=1)))
print(y_test.values)
print(type(y_test.values))
action_matrix = confusion_matrix(y_test.values.tolist(), np.rint(action_pred.flatten()).tolist())


y = movie_data[1]['Comedy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
comedy_pred = comedy_model.predict(X_test)
print(comedy_pred)
comedy_matrix = confusion_matrix(y_test.values.tolist(), np.rint(comedy_pred.flatten()).tolist())
print(action_matrix)
print(comedy_matrix)

#%% Create Graph
plt.title('Actual')
seaborn.set(font_scale=1.1)#for label size
df_action = pd.DataFrame(action_matrix, index = ['Action', 'N Action'],
                         columns=['Action', 'N Action'])
sb = seaborn.heatmap(df_action, annot=True, fmt='g').xaxis.set_ticks_position('top')
plt.ylabel('Predicted')
plt.show()

plt.title('Actual')
df_comedy = pd.DataFrame(comedy_matrix, index = ['Comedy', 'Not Com'],
                         columns=['Comedy', 'Not Com'])
sb = seaborn.heatmap(df_comedy, annot=True, fmt='g').xaxis.set_ticks_position('top')
plt.ylabel('Predicted')
plt.show()
