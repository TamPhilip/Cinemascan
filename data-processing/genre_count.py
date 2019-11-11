import pandas as pd
import re
import csv
import numpy as np
import os

path = os.path.abspath(os.curdir)

class GenreCount:



    def __init__(self):
        print("Hl")

    @staticmethod
    def count_genre():
        print()


movie_data = pd.read_csv('{}/../csv-data/movie-data.csv'.format(path))

print(" \n BEFORE DATA CLEAN \n")
print(len(movie_data))

for x in movie_data:
    count = len(movie_data[movie_data[x] == 1])
    if count > 4000:
        print("{}: {}".format(x, count))

# Remove existing Crime Column
movie_data.drop(columns=['Crime'], axis=1, inplace=True)

# Choose Action, Comedy, Horror, Crime, Romance
movie_data.rename(columns={'Romance Film': 'Romance', 'Crime Fiction': 'Crime'}, inplace=True)

movie_data = movie_data[['Title',
                         'Summary',
                         'Action',
                         'Adventure',
                         'Comedy',
                         'Drama',
                         'Thriller',
                         'Horror',
                         'Romance',
                         'Crime']]

#select data

movie_data = movie_data[(movie_data['Action'] == 1) |
                        (movie_data['Comedy'] == 1) |
                        (movie_data['Drama'] == 1) |
                        (movie_data['Adventure'] == 1) |
                        (movie_data['Thriller'] == 1) |
                        (movie_data['Horror'] == 1) |
                        (movie_data['Romance'] == 1) |
                        (movie_data['Crime'] == 1)]

print(len(movie_data))

print(" \n SELECT DATA \n")


def check(r, a, dr, co, th, ho, cr):
    genres = ['Action',
              'Comedy',
              'Drama',
              'Thriller',
              'Horror',
              'Romance',
              'Crime']
    movie = movie_data[ (movie_data['Romance'] == r) & (movie_data['Action'] == a) & (movie_data['Drama'] == dr) & (movie_data['Comedy'] == co) & (movie_data['Thriller'] == th) & (movie_data['Horror'] == ho) & (movie_data['Crime'] == cr)
    ]
    if r == 0:
        genres.remove('Romance')
    if a ==0:
        genres.remove('Action')
    if dr == 0:
        genres.remove('Drama')
    if co == 0:
        genres.remove('Comedy')
    if th == 0:
        genres.remove('Thriller')
    if ho == 0:
        genres.remove('Horror')
    if cr == 0:
        genres.remove('Crime')
    if len(movie) > 100:
        print(" {} : {}".format(genres, len(movie)))

for r in range(2):
    for a in range(2):
        for dr in range(2):
            for co in range(2):
                for th in range(2):
                    for ho in range(2):
                        for cr in range(2):
                            check(r,a,dr,co,th,ho,cr)

# action_movies.to_csv(path_or_buf='{}/csv-data/action-data.csv'.format(path))
# comedy_movies.to_csv(path_or_buf='{}/csv-data/comedy-data.csv'.format(path))
# drama_movies.to_csv(path_or_buf='{}/csv-data/drama-data.csv'.format(path))
# thriller_movies.to_csv(path_or_buf='{}/csv-data/thriller-data.csv'.format(path))
# horror_movies.to_csv(path_or_buf='{}/csv-data/horror-data.csv'.format(path))

print(movie_data.columns)
movie_data['Summary'] = movie_data['Summary'].str.lower().replace('[?!]', '.', regex=True)
movie_data['Summary'] = movie_data['Summary'].str.lower().replace("[\\\{\}\[\]\)\(\*\:]+|(\W+plot\W+)|(['’]s)|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|(cite (web|news|book))", '', regex=True)
movie_data['Summary'] = movie_data['Summary'].str.lower().replace("(\|.*>)|(<.*>?)|(quote box)|(plot\W)|^(plot|introduction|beginning)\W|.citation$|.fact$|(\.\w*\W+\w*)$|[>$'’]|(\w*[;&|]+\w*)+|\s(\d)+", '', regex=True)
movie_data['Summary'] = movie_data['Summary'].str.lower().replace('["\'#$%&()*+,-/:;<=>@[\\]^_`{|}~\t\n]',' ', regex=True)
movie_data['Summary'] = movie_data['Summary'].str.lower().replace('[-,"]', ' ', regex=True)

# np.savetxt("summaries.txt", list(enumerate(movie_data['Summary'])), delimiter=" ", newline = "\n \n \n", fmt="%s")

movie_data.to_csv('final_data.csv')
