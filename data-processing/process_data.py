import pandas as pd
import json
import os

path = os.path.abspath(os.curdir)

class Preprocessing:

    """
    Helper class for Preprocessing Data
    """

    @staticmethod
    def extract_unique_list_from_str_dict(df, column):
        """

        Data looks like:
        "{"/m/01jfsb": "Thriller", "/m/06n90": "Science Fiction", "/m/03npn": "Horror", "/m/03k9fj": "Adventure"}"

        :param df:DataFrame Df containing rows that contain string values in a dictionary format
        :param column:str Column name

        :return: list
         list of unique values of the column
        """
        s = set()
        print(df[column][0])
        for x in range(len(df[column])):
            values = list(json.loads(df[column][x]).values())
            for value in values:
                s.add(value)
        return list(s)

    @staticmethod
    def create_row(df, ind, id, word_list, plot_df, genre_list):
        dictionary = {"Title": df["title"][ind], "Summary": word_list[plot_df["wikiid"].tolist().index(id)]}
        zeros = [0] * len(genre_list)
        genre_dict = dict(zip(genre_list, zeros))
        to_return = {**genre_dict, **dictionary}
        for genre in list(json.loads(df['genres'][ind]).values()):
            to_return[genre] = 1
        return to_return


    @staticmethod
    def create_row(column_names, values, dv, dv_values):
        """
        @Precondition: dv is unique list

        For supervised learning, set all of the dv's to 1 for the specified dependent variables

        :param columns_names: [str], column names (ordered with values)
        :param values: [any] - List of values of columns
        :param dv: [str] (unique list of dependent_variables)
        :param dv_values: [str] (list of dependent variables)
        :return:
            dictionary that will appended  to  a DataFrame
        """
        assert (len(dv) == len(set(dv))), "Not  a unique list"

        dictionary = {}
        for index, column in  enumerate(column_names):
            dictionary[column] = values[index]

        row = {**dict(zip(dv, [0] * len(dv))), **dictionary}
        for v in dv_values:
            row[v] = 1
        return row


    @staticmethod
    def iterate_through_all(n_df, df, word_list, genre_list, plot_df):
        """

        :param n_df:
        :param df:
        :param word_list:
        :param genre_list:
        :param plot_df:
        :return:
        """
        for ind, id in enumerate(df['wikiid']):
            print(ind)
            if id in plot_df['wikiid'].tolist():
               n_df = n_df.append(Preprocessing.create_row(["Title","Summary"],[df["title"][ind],word_list[plot_df["wikiid"].tolist().index(id)]], genre_list, list(json.loads(df['genres'][ind]).values())), ignore_index=True)
        return n_df

    @staticmethod
    def merge_data(df_1, df_2, columns):
        """

        :param df_1:
        :param df_2:
        :param columns:
        :return:
        """


if __name__ == '__main__':
    df = pd.read_csv('{}/../movie-data/corpus/movie.metadata.tsv'.format(path), delimiter='\t', encoding='utf-8')
    print(len(df))
    plot_df = pd.read_csv('{}/../movie-data/corpus/plot_summaries.txt'.format(path), delimiter='\t')
    print(len(plot_df))
    word_list = [x.lower().replace(",", "") for x in plot_df["summary"]]
    print(word_list)

    # list of genres
    # genre_list = Preprocessing.extract_unique_list_from_str_dict(df, "genres")
    # print(genre_list)
    # #
    # list_title = genre_list
    # list_title.append("Title")
    # list_title.append("Summary")
    #
    # n_df = pd.DataFrame(columns=list_title)
    # # #
    # n_df = Preprocessing.iterate_through_all(n_df, df, word_list, genre_list, plot_df)
    # print(n_df.columns)
    #
    # print(len(n_df))
    # print(n_df)
    # n_df.to_csv(path_or_buf='/csv-data/movie-data.csv')
    #






# #stats imdb
# df_im = pd.read_csv('movies_genres.csv', delimiter='\t')
# imdb_genres = df_im.drop(['plot', 'title', 'Sci-Fi','Documentary', 'Reality-TV', 'Animation'], axis=1)
# counts = []
# categories = list(imdb_genres.columns.values)
# for i in categories:
#     counts.append((i, imdb_genres[i].sum()))
# df_stats_imdb = pd.DataFrame(counts, columns=['genre', '#movies'])
#
# df_stats_imdb = df_stats_imdb[df_stats_imdb['#movies'] > 8000]
# df_stats_imdb
# # df_stats_imdb['genre'].values
# corpus_genres = df_co.drop(['Title', 'Summary', 'Horror'], axis=1)
# counts = []
# categories = list(corpus_genres.columns.values)
# for i in categories:
#   counts.append((i, corpus_genres[i].sum()))
# df_stats_corpus = pd.DataFrame(counts, columns=['genre', '#movies'])
# #Stats corpus
#
# df_co = pd.read_csv('movie-data-cleaned.csv')
# df_co.drop(['Unnamed: 0'], axis=1, inplace=True)
# corpus_genres = df_co.drop(['Title', 'Summary', 'Horror'], axis=1)
# counts = []
# categories = list(corpus_genres.columns.values)
# for i in categories:
#   counts.append((i, corpus_genres[i].sum()))
# df_stats_corpus = pd.DataFrame(counts, columns=['genre', '#movies'])
#
# #  Actually merging
# df_im = df_im.drop(['Sci-Fi','Documentary', 'Reality-TV', 'Animation'], axis=1)
# df_co = df_co.drop(['Horror'], axis=1)
# df_co.rename(index=str, columns={"Title": "title", "Summary": "plot"}, inplace=True)
#
# print(df_im.head())
# print(df_co.head())
# df_final = df_im.append(df_co)
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
#
# df_final = pd.concat([df_final_plot, df_final_values], axis=1, join='inner')
#
# print(df_final)

