#%%
import pandas as pd
import json
import os

class Preprocessing:
    optional  = object()

    """
    Helper class for Preprocessing Data
    """

    @staticmethod
    def extract_str_dict_df(df, column):
        """

        To parse data that have rows that look like:
        "{"/m/01jfsb": "Thriller", "/m/06n90": "Science Fiction", "/m/03npn": "Horror", "/m/03k9fj": "Adventure"}"

        :param df:DataFrame Df containing rows that contain string values in a dictionary format
        :param column:str Column name

        :return: list
         a tuple consisting of a`list of unique values of the column, list_of_values
        """
        s = set()
        lists = []
        for x in range(len(df[column])):
            values = list(json.loads(df[column][x]).values())
            lists.append(values)
            for value in values:
                s.add(value)

        unique_list = list(s)
        return unique_list, lists


    @staticmethod
    def create_row(c_v, values, c_dv, dv_values):
        """
        @Precondition: dv is unique list

        Essentially one hot encodes the dependent values

        # Any other relevant data
        :param c_v: [str], column names (ordered with values)
        :param values: [any] - List of values of columns

        # Categorical data
        :param c_dv: [str] (unique list of dependent_variables)
        :param dv_values: [str] (list of dependent variables)
        :return:
            dictionary that will appended  to a DataFrame
        """
        assert (len(c_dv) == len(set(c_dv))), "Not  a unique list"

        # Any other relevant data
        dictionary = dict(zip(c_v, values))

        # One hot encoding
        dictionary_2 = dict(zip(c_dv, [0] * len(c_dv)))
        for v in dv_values:
            dictionary_2[v] = 1

        row = {**dictionary_2, **dictionary}
        return row


    @staticmethod
    def create_cleaned_df(meta_df, plot_df, columns, genre_list):
        """
        @precondition: Must be for the CMU Corpus Data Frame

        :param meta_df: DataFrame for movie.metadata.tsv
        :param plot_df: DataFrame for plot-summaries.txt
        :param columns: Columns for the cleaned DataFrame
        :param genre_list: List of (genres for each movie)
        :return:
            DataFrame, but cleaned and arranged poperly
        """

        n_df = pd.DataFrame(columns=columns)
        plot_ids = plot_df['wikiid'].tolist()
        plot_by_id = [x.lower().replace(",", "") for x in plot_df["summary"]]
        for ind, id in enumerate(meta_df['wikiid']):
            if id in plot_ids:
                row = Preprocessing.create_row(["Title","Summary"],[meta_df["title"][ind], plot_by_id[plot_by_id.index(id)]], columns, genre_list[ind])
                n_df = n_df.append(row, ignore_index=True)
        return n_df

    @staticmethod
    def merge_data(df_1, df_2, co_to_keep, rn_co_1=optional, rn_co_2=optional):
        """

        Formats the columns and merges the dataframes based on those columns

        :param df_1: Data Frame 1
        :param df_2: Data Frame 2
        :param co_to_keep: [str] - List of columns to keep
        :param rn_co_1: Dictionary - Renamed columns for DF1
        :param rn_co_2: Dictionary - Renamed columns for DF2
        :return:
            merged DataFrame
        """
        if rn_co_1 != Preprocessing.optional:
            df_1.rename(columns=rn_co_1, inplace=True)
        if rn_co_2 != Preprocessing.optional:
            df_2.rename(columns=rn_co_2, inplace=True)
        return pd.concat(df_1[co_to_keep], df_2[co_to_keep])


    # TODO: Count the genres for imdb dataset
    @staticmethod
    def count_genres_imdb(df):
        """
        

        
        :param df:
        :return:
        """

    @staticmethod
    def keep_columns(df, bound, keep_columns):
        """

        Returns the columns with at least 5000 movies of that genre

        :param df: Data Frame
        :param bound: Int - Determines the amount needed to be added as a column to keep
        :param keep_columns: List(Strings) - List of Strings to keep
        :return: columns : List(Strings) - List of columns to keep
        """

        columns = keep_columns
        for genre in df:
            count = len(df[df[genre] == 1])
            if count > bound:
                columns.append(genre)
        return columns


    @staticmethod
    def clean_text_for_training(df, column):
        """
        Strips off all special characters and lower cases all characters from a column of a dataframe.

        :param df: Data Frame
        :param column: String - Column to strip
        :return: Data Frame
        """
        df_c = df.copy()
        df_c[column] = df_c[column].str.lower().replace('[?!]', '.', regex=True)
        df_c[column] = df_c[column].str.lower().replace(
            "[\\\{\}\[\]\)\(\*\:]+|(\W+plot\W+)|(['’]s)|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|(cite (web|news|book))",
            '', regex=True)
        df_c[column] = df_c[column].str.lower().replace(
            "(\|.*>)|(<.*>?)|(quote box)|(plot\W)|^(plot|introduction|beginning)\W|.citation$|.fact$|(\.\w*\W+\w*)$|[>$'’]|(\w*[;&|]+\w*)+|\s(\d)+",
            '', regex=True)
        df_c[column] = df_c[column].str.lower().replace('["\'#$%&()*+,-/:;<=>@[\\]^_`{|}~\t\n]', ' ',
                                                        regex=True)
        df_c[column] = df_c[column].str.lower().replace('[-,"]', ' ', regex=True)
        return df_c
