import pandas as pd
import re
import csv
import numpy as np
import os
from src.data_processing.preprocessing import Preprocessing
from src.data_processing.counting import Counting

#%% Get path
path = os.path.abspath(os.curdir)

#%%

movie_data = pd.read_csv('{}/data/csv-data/movie-data.csv'.format(path))

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

movie_data = movie_data[['Title', 'Summary', 'Action', 'Adventure', 'Comedy', 'Drama', 'Thriller', 'Horror', 'Romance', 'Crime']]

# Keep all re

movie_data = movie_data[(movie_data['Action'] == 1) |
                        (movie_data['Comedy'] == 1) |
                        (movie_data['Drama'] == 1) |
                        (movie_data['Adventure'] == 1) |
                        (movie_data['Thriller'] == 1) |
                        (movie_data['Horror'] == 1) |
                        (movie_data['Romance'] == 1) |
                        (movie_data['Crime'] == 1)]

#%%

genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Thriller', 'Crime', 'Adventure']
Counting.check_counts_by_genre(movie_data, genres, 100)


#%% Stats imdb
df_im = pd.read_csv('{}/data/movie-data/imdb/movies_genres.csv'.format(path), delimiter='\t')


Counting.get_count_by_genre(df_im, ['plot', 'title', 'Sci-Fi', 'Documentary', 'Reality-TV', 'Animation'], 0)

#%% Stats corpus

df_co = pd.read_csv('{}/data/csv-data/movie-data.csv'.format(path))
Counting.get_count_by_genre(df_co, ['Title', 'Summary', 'Horror'], 500)

#%% TODO: Move to script

movie_data = Preprocessing.clean_text_for_training(movie_data, 'Summary')

movie_data.to_csv('final_data.csv')