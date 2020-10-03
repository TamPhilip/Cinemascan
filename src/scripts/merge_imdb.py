"""

Script: Merges the movie-genre.csv file with the idmb dataset => create final_data.csv

"""

#%%
import pandas as pd
import os
from src.data_processing.preprocessing import Preprocessing
path = os.path.abspath(os.curdir)

#%% Import csv files

# IMDB and Corpus
df_im = pd.read_csv('{}/data/movie-data/imdb/movies_genres.csv'.format(path), delimiter='\t')
df_co = pd.read_csv('{}/data/csv-data/movie-data.csv'.format(path))

#%%

print("------------ Corpus ------------")
columns_co = Preprocessing.keep_columns(df_co, 4000, ['Title', 'Summary'])
print(len(columns_co))
print(columns_co)

print("------------ IMDB ------------")
columns_imdb = Preprocessing.keep_columns(df_im, 4000, ['title', 'plot'])
print(len(columns_imdb))
print(columns_imdb)

#%% Rename so Title and Summary is consistent

df_im.rename(index=str, columns={"Title": "title", "Summary": "plot"}, inplace=True)


# %%
df_im = df_im.drop(['Sci-Fi','Documentary', 'Reality-TV', 'Animation'], axis=1)
df_co = df_co.drop(['Horror'], axis=1)


#%%

print(df_im.head())
print(df_co.head())
#
# print(df_final.columns)
# df_final.fillna(int(0), inplace=True)
#
# df_final_plot = df_final[['plot']]
# df_final_values = df_final[['Action',
#                             'Adventure',
#                             'Comedy',
#                             'Crime',
# #                             'Drama',
#                             'Family',
#                             'Mystery',
#                             'Romance',
#                             'Thriller']].astype(int)
#
# df_final_plot['plot'] = df_final_plot['plot'].str.lower().replace('["\'#$%&()*+,-/:;<=>@[\\]^_`{|}~\t\n]',' ', regex=True)

# df_final = pd.concat([df_final_plot, df_final_values], axis=1, join='inner')
#
# print(df_final)

