import pandas as pd
import json
import os

path = os.path.abspath(os.curdir)

df = pd.read_csv('{}/movie-data/movie.metadata.tsv'.format(path),delimiter='\t',encoding='utf-8')

def get_genre_list(df):
    set_title = set()
    for x in range(len(df['genres'])):
        genres = list(json.loads(df['genres'][x]).values())
        for genre in genres:
            set_title.add(genre)
    print(set_title)
    print(len(set_title))
    return list(set_title)


def create_row(df, ind, id, word_list, plot_df, genre_list):
    dictionary = {"Title": df["title"][ind], "Summary": word_list[plot_df["wikiid"].tolist().index(id)]}
    zeros = [0] * len(genre_list)
    genre_dict = dict(zip(genre_list, zeros))
    to_return = {**genre_dict, **dictionary}
    for genre in list(json.loads(df['genres'][ind]).values()):
        to_return[genre] = 1
    return to_return


def iterate_through_all(n_df, df, word_list, genre_list, plot_df):
    for ind, id in enumerate(df['wikiid']):
        print(ind)
        if id in plot_df['wikiid'].tolist():
           n_df = n_df.append(create_row(df, ind, id, word_list, plot_df, genre_list), ignore_index=True)
    return n_df


def iterate_through_most(n_df, df, word_list, genre_list, plot_df):
    for x, id in enumerate(plot_df['wikiid'].tolist()):
        if list(df['wikiid']).__contains__(id):
            print(x)
            ind = list(df['wikiid']).index(id)
            n_df = n_df.append(create_row(df, ind, id, word_list, plot_df, genre_list), ignore_index=True)
    return n_df

#example
# print(df.head())
#shape
# print(df.shape)
#dimensions
# print(df.ndim)
# columns names
# print(df.columns)
# wikiid
# print(df['wikiid'])
# title
# print(df['title'])
# genres -> FIX

print(df['genres'])

plot_df = pd.read_csv('{}/movie-data/plot_summaries.txt'.format(path), delimiter='\t')
#
# print(plot_df.head(10))
#
word_list = [x.lower().replace(",","") for x in plot_df["summary"]]
# #
# print(len(word_list))
# print(plot_df["wikiid"].tolist())
#

genre_list = get_genre_list(df)

list_title = genre_list
list_title.append("Title")
list_title.append("Summary")

n_df = pd.DataFrame(columns= list_title)

n_df = iterate_through_all(n_df, df, word_list, genre_list,plot_df)
print(n_df.columns)

print(len(n_df))
print(n_df)
n_df.to_csv(path_or_buf='{}/csv-data/movie-data.csv'.format(path))
#





