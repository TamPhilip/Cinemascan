#%%
"""
Analyze model:
Meant to analyze each models and their p[erformance
"""

import h5py
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
import os
from keras.models import load_model
path = os.path.abspath(os.curdir)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#%%
df = pd.read_csv('{}/csv-data/movie-data.csv'.format(path))
df.drop(columns='Unnamed: 0', inplace=True)
# action_model = load_model('{}/model_version/n_most/action_model.h5'.format(path))
# adventure_model = load_model('{}/model_version/n_most/adventure_model.h5'.format(path))
# comedy_model = load_model('{}/model_version/n_most/comedy_model.h5'.format(path))
# crime_model = load_model('{}/model_version/n_most/crime_model.h5'.format(path))
# family_model = load_model('{}/model_version/n_most/family_model.h5'.format(path))
# mystery_model = load_model('{}/model_version/n_most/mystery_model.h5'.format(path))
# romance_model = load_model('{}/model_version/n_most/romance_model.h5'.format(path))
# thriller_model = load_model('{}/model_version/n_most/thriller_model.h5'.format(path))
#%%
print(df.head())

#%%
features = df['plot'].values

n_most_common_words = 10000
max_len = 500
tokenizer = Tokenizer(num_words=n_most_common_words, lower=True)

tokenizer.fit_on_texts(features)
sequences = tokenizer.texts_to_sequences(features)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
# print(word_index)
X = pad_sequences(sequences, maxlen=500)
# print(X)

#%%
models = {"Action": action_model,
		  "Adventure": adventure_model,
		  'Comedy': comedy_model,
		  "Crime": crime_model,
		  "Family": family_model,
		  "Mystery": mystery_model,
		  "Romance": romance_model,
		  "Thriller": thriller_model}

for genre, model in models.items():
	y = df[genre]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
	prediction = model.predict(X_test)
	cf_m = confusion_matrix(y_test.values.tolist(), np.rint(prediction.flatten()).tolist())
	plt.title('Actual')
	seaborn.set(font_scale=1.1)  # for label size
	rep = pd.DataFrame(cf_m, index=['N {}'.format(genre), genre],
							 columns=['N {}'.format(genre), genre])
	sb = seaborn.heatmap(rep, annot=True, fmt='g').xaxis.set_ticks_position('top')
	plt.ylabel('Predicted')
	plt.xlabel('Bidirectional LSTM')
	plt.show()

#%%
print(path)
df_co = pd.read_csv('{}/csv-data/movie-data-cleaned.csv'.format(path))
df_co.drop(['Unnamed: 0'], axis=1, inplace=True)
df_im = pd.read_csv('{}/csv-data/movies_genres.csv'.format(path), delimiter='\t')

df_im.head()

imdb_genres = df_im.drop(['plot', 'title', 'Sci-Fi','Documentary', 'Reality-TV', 'Animation'], axis=1)
counts = []
categories = list(imdb_genres.columns.values)
for i in categories:
    counts.append((i, imdb_genres[i].sum()))
df_stats_imdb = pd.DataFrame(counts, columns=['genre', '#movies'])

df_stats_imdb = df_stats_imdb[df_stats_imdb['#movies'] > 8000]
df_stats_imdb
# df_stats_imdb['genre'].values

df_co.head()

corpus_genres = df_co.drop(['Title', 'Summary', 'Horror'], axis=1)
counts = []
categories = list(corpus_genres.columns.values)
for i in categories:
  counts.append((i, corpus_genres[i].sum()))
df_stats_corpus = pd.DataFrame(counts, columns=['genre', '#movies'])
df_stats_corpus

cs = []

for index, category in enumerate(df_stats_imdb['genre']):
  current_index_b = 0
  for index_b, category_b in enumerate(df_stats_corpus['genre']):
    if category == category_b:
      current_index_b = index_b
      cs.append((category, df_stats_corpus['#movies'].values[index_b] + df_stats_imdb['#movies'].values[index]))
  if not (category, df_stats_corpus['#movies'].values[current_index_b] + df_stats_imdb['#movies'].values[index]) in cs:
      cs.append((category, df_stats_imdb['#movies'].values[index]))

df_stats = pd.DataFrame(cs, columns=['genre', '#movies'])
df_stats