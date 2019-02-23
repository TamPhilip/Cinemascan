import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from nltk.corpus import stopwords
from keras.layers import Bidirectional

stop_words = set(stopwords.words('english'))
stop_words = {x.replace("'","") for x in stop_words if re.search("[']", x.lower())}

genres = ['Action',
          'Comedy',
          'Drama',
          'Thriller',
          'Horror',
          'Romance',
          'Crime']

path = os.path.abspath(os.curdir)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
movie_data = pd.read_csv('{}/../csv-data/movie-data-cleaned.csv'.format(path))

movie_data = np.split(movie_data, [3], axis=1)
movie_data[0].drop(columns='Unnamed: 0', inplace=True)
features = movie_data[0]['Summary'].values.astype('U')
vectorizer = TfidfVectorizer(stop_words=stop_words, binary=True)
X = vectorizer.fit_transform(features).toarray()
X = pad_sequences(X)

for genre in genres:

    labels = movie_data[1][genre]

# n_most_common_words = 10000
# max_len = 500
# tokenizer = Tokenizer(num_words=n_most_common_words, lower=True)
# tokenizer.fit_on_texts(features)
# sequences = tokenizer.texts_to_sequences(features)
# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))

# print(X.shape[0])
# X_test = vectorizer.transform(X_test)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.15, random_state=42)

    epochs = 10
    emb_dim = 128
    batch_size = 256

    print((X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    model = Sequential()
    print(X.shape[0])
    model.add(Embedding(X.shape[0], emb_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.7))
    model.add(Bidirectional(LSTM(64, recurrent_dropout=0.7)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss',patience=7, min_delta=0.0001)])

