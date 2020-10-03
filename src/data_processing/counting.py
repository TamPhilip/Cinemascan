import pandas as pd

class Counting:
    optional  = object()

    """
    Helper class for Counting Data
    """

    @staticmethod
    def check_counts_by_genre(df, columns, bound):
        """

        Prints out all of the combinations of genres above a certain bound

        :param df: Data Frame - DF of Genres (values: 0 or 1)
        :param columns: List (string) - Columns of Genres
        :param bound: Int - Cut out all values of length below bound
        :return: None
        """
        counts = df.groupby(columns, as_index=False).count()

        counts = counts[counts['Summary'] > bound]

        def check_row(row):
            genres = []
            row_columns = row[['Action', 'Adventure', 'Comedy', 'Drama', 'Thriller', 'Horror', 'Romance', 'Crime']]
            for column, value in row_columns.iteritems():
                if value == 1:
                    genres.append(column)
            print(" {} : {}".format(genres, row['Summary']))

        for index, row in counts.iterrows():
            check_row(row)

    @staticmethod
    def get_count_by_genre(df, drop_columns, bound):
        """

        :param df:
        :param drop_columns:
        :param bound:
        :return:
        """
        genres = df.drop(drop_columns, axis=1)
        counts = []
        categories = list(genres.columns.values)
        for i in categories:
            counts.append((i, genres[i].sum()))

        stats = pd.DataFrame(counts, columns=['genre', '#movies'])

        stats = stats[stats['#movies'] > bound]
        print(stats)

