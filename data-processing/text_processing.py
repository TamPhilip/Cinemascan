#%% Run Imports
import pandas as pd
import re
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import os

#%% Run preprocessin
path = os.path.abspath(os.curdir)
stop_words = set(stopwords.words('english'))
stop_words = {x.replace("'","") for x in stop_words if re.search("[']", x.lower())}

movie_data = pd.read_csv('../csv-data/movie-data-cleaned.csv')
print(len(movie_data))

msk = np.random.rand(len(movie_data)) < 0.8

genres = ['Action',
          'Comedy',
          'Adventure',
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

#%% Run Predict
def predict_model(type, model, vectorizer, train_labels, train_features, test_labels, test_features):
    print("\n {} \n".format(type))
    pipeline = Pipeline([
        ('vec', vectorizer),
        ('clf', model),
    ])

    predictions = {}
    for genre in genres:
        print('... Processing {}'.format(genre))
        # train the model using X_dtm & y
        pipeline.fit(train_features, np.array(train_labels[genre]).astype('int'))
        # compute the testing accuracy
        prediction = pipeline.predict(test_features)
        accuracy = accuracy_score(np.array(test_labels[genre]).astype('int'), prediction)
        cm = confusion_matrix(np.array(test_labels[genre]).astype('int'), prediction)
        print('Test accuracy is {}'.format(accuracy))
        predictions[genre] = (accuracy, cm)
    return predictions

#TFIdVectorizer = CountVectorizer + TfidTransformer (Normalizing)

#%% Run Predictions
type = "Logistic Regression TfidVectorizer"
logistic_results = predict_model(type, OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1),
                                        TfidfVectorizer(stop_words=stop_words),
                                        train_labels,
                                        train_features,
                                        test_labels,
                                        test_features)

type = "Support Vector Machine TfidVectorizer"
svm_results = predict_model(type, OneVsRestClassifier(LinearSVC(), n_jobs=1),
                                        TfidfVectorizer(stop_words=stop_words),
                                        train_labels,
                                        train_features,
                                        test_labels,
                                        test_features)

type = "Naive Bayes TfidVectorizer"
nv_results = predict_model(type, OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None), n_jobs=1),
                                        TfidfVectorizer(stop_words=stop_words),
                                        train_labels,
                                        train_features,
                                        test_labels,
                                        test_features)
#%% Run Data analysis