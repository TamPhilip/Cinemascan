import pandas as pd
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import os

path = os.path.abspath(os.curdir)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words = {x.replace("'","") for x in stop_words if re.search("[']", x.lower())}

movie_data = pd.read_csv('{}/csv-data/movie-data-cleaned.csv'.format(path))
print(len(movie_data))

msk = np.random.rand(len(movie_data)) < 0.9

genres = ['Action',
          'Comedy',
          'Drama',
          'Thriller',
          'Horror',
          'Romance',
          'Crime']

train = movie_data[msk]
train = np.split(train, [3], axis=1)
train[0].drop(columns='Unnamed: 0', inplace=True)
train_features = train[0]['Summary'].values.astype('U')
train_labels = train[1]

test = movie_data[~msk]
test = np.split(test, [3], axis=1)
test[0].drop(columns='Unnamed: 0', inplace=True)
test_features = test[0]['Summary'].values.astype('U')
test_labels = test[1]

def predict_model(type, model, vectorizer, train_labels, train_features, test_labels, test_features):
    print("\n {} \n".format(type))
    pipeline = Pipeline([
        ('vec', vectorizer),
        ('clf', model),
    ])

    for genre in genres:
        print('... Processing {}'.format(genre))
        # train the model using X_dtm & y
        pipeline.fit(train_features, np.array(train_labels[genre]).astype('int'))
        # compute the testing accuracy
        prediction = pipeline.predict(test_features)
        print('Test accuracy is {}'.format(accuracy_score(np.array(test_labels[genre]).astype('int'), prediction)))


#TFIdVectorizer = CountVectorizer + TfidTransformer (Normalizing)

type = "Logistic Regression TfidVectorizer"
predict_model(type, OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1),
                                        TfidfVectorizer(stop_words=stop_words),
                                        train_labels,
                                        train_features,
                                        test_labels,
                                        test_features)

type = "Support Vector Machine TfidVectorizer"
predict_model(type, OneVsRestClassifier(LinearSVC(), n_jobs=1),
                                        TfidfVectorizer(stop_words=stop_words),
                                        train_labels,
                                        train_features,
                                        test_labels,
                                        test_features)

type = "Naive Bayes TfidVectorizer"
predict_model(type, OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None), n_jobs=1),
                                        TfidfVectorizer(stop_words=stop_words),
                                        train_labels,
                                        train_features,
                                        test_labels,
                                        test_features)
