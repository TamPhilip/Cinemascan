"""

Script: Creates the movie-data.csv file

"""

import os
import pandas as pd
import json
from src.data_processing.preprocessing import Preprocessing


path = os.path.abspath(os.curdir)
# load data from CMU Corpus
meta_df = pd.read_csv('{}/data/movie-data/corpus/movie.metadata.tsv'.format(path), sep='\t', encoding='utf-8')
plot_df = pd.read_csv('{}/data/movie-data/corpus/plot_summaries.txt'.format(path), delimiter='\t')

# list of genres
columns, genre_list = Preprocessing.extract_str_dict_df(meta_df, "genres")

# Appends the columns for title and summary
columns.append("Title")
columns.append("Summary")

# Adds the data frame from the plot data and movie metadata (Genres, title)
n_df = Preprocessing.iterate_through_all(meta_df=meta_df, plot_df=plot_df, columns=columns, column_v=genre_list)

# Save dataframe to csv
n_df.to_csv(path_or_buf="{}/data/csv-data/movie-data.csv".format(path), index=False)

