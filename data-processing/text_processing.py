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
import seaborn
import matplotlib.pyplot as plt
from sklearn.externals import joblib
#%% Run preprocessin
path = os.path.abspath(os.curdir)
stop_words = set(stopwords.words('english'))
stop_words = {x.replace("'","") for x in stop_words if re.search("[']", x.lower())}
movie_data = pd.read_csv('{}/../csv-data/final_data.csv'.format(path))
print(len(movie_data))

msk = np.random.rand(len(movie_data)) < 0.8

genres = ['Action',
              'Comedy',
              # 'Drama',
              'Thriller',
              'Family',
              'Adventure',
              'Mystery',
              'Romance',
              'Crime'
          ]

train = movie_data[msk]
train = np.split(train, [2], axis=1)
train[0].drop(columns='Unnamed: 0', inplace=True)
train_features = train[0]['plot'].values.astype('U')
print(train[1])
train_labels = train[1]

test = movie_data[~msk]
test = np.split(test, [2], axis=1)
test[0].drop(columns='Unnamed: 0', inplace=True)
test_features = test[0]['plot'].values.astype('U')
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
        joblib.dump(pipeline,'{}_{}.pkl'.format(type, genre))
    joblib.dump(pipeline, '{}.pkl'.format(type))
    return predictions

#TFIdVectorizer = CountVectorizer + TfidTransformer (Normalizing)

#%% Run Predictions
type = "Logistic_Regression_TfidVectorizer"
logistic_results = predict_model(type, OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1),
                                        TfidfVectorizer(stop_words=stop_words),
                                        train_labels,
                                        train_features,
                                        test_labels,
                                        test_features)

type = "Support_Vector_Machine_TfidVectorizer"
svm_results = predict_model(type, OneVsRestClassifier(LinearSVC(), n_jobs=1),
                                        TfidfVectorizer(stop_words=stop_words),
                                        train_labels,
                                        train_features,
                                        test_labels,
                                        test_features)

type = "Naive_Bayes_TfidVectorizer"
nv_results = predict_model(type, OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None), n_jobs=1),
                                        TfidfVectorizer(stop_words=stop_words),
                                        train_labels,
                                        train_features,
                                        test_labels,
                                        test_features)
#%% Run Data analysis
plt.title('Actual')
seaborn.set(font_scale=1.1)#for label size
df_action = pd.DataFrame(nv_results['Action'][1], index = ['Action', 'N Action'],
                         columns=['Action', 'N Action'])
sb = seaborn.heatmap(df_action, annot=True, fmt='g').xaxis.set_ticks_position('top')
plt.ylabel('Predicted')
plt.xlabel('Logistic Regression')
plt.show()

plt.title('Actual')
df_comedy = pd.DataFrame(nv_results['Comedy'][1], index = ['Comedy', 'Not Com'],
                         columns=['Comedy', 'Not Com'])
sb = seaborn.heatmap(df_comedy, annot=True, fmt='g').xaxis.set_ticks_position('top')
plt.ylabel('Predicted')
plt.xlabel('Logistic Regression')
plt.show()