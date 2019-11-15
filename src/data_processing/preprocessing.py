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


    # TODO: Count the genres for each
    @staticmethod
    def count_genre(df):
        """
        
        :param df: 
        :return: 
        """
        print()

