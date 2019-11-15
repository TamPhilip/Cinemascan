"""

Script: Merges the movie-genre.csv file with the idmb dataset => create final_data.csv

"""

#%%
import pandas as pd
import os
path = os.path.abspath(os.curdir)

if __name__ == '__main__':


    # #  Actually merging
    df_im = pd.read_csv('{}/data/movie-data/imdb/movies_genres.csv'.format(path), delimiter='\t')
    df_co = pd.read_csv('{}/data/csv-data/movie-data.csv'.format(path))

    # df_im = df_im.drop(['Sci-Fi','Documentary', 'Reality-TV', 'Animation'], axis=1)
    # df_co = df_co.drop(['Horror'], axis=1)
    df_im.rename(index=str, columns={"Title": "title", "Summary": "plot"}, inplace=True)
    print("---------------")
    print(len(df_co[df_co['Romantic drama'] == 1]))
    print(len(df_co[(df_co['Romantic drama'] == 1) & (df_co['Romance Film'] == 1) & (df_co['Drama'] == 1)]))
    print(len(df_co[(df_co['Romance Film'] == 1) & (df_co['Drama'] == 1)]))

    print("---------------")
    print(len(df_co[df_co['Romantic comedy'] == 1]))
    print(len(df_co[(df_co['Romantic comedy'] == 1) & (df_co['Romance Film'] == 1) & (df_co['Comedy'] == 1)]))
    print(len(df_co[(df_co['Romance Film'] == 1) & (df_co['Comedy'] == 1)]))

    print("------------ Corpus ------------")
    for x in df_co:
        count = len(df_co[df_co[x] == 1])
        if count > 1000:
            print("{}: {}".format(x, count))


    print("------------ IMDB ------------")

    for x in df_im:
        count = len(df_im[df_im[x] == 1])
        if count > 1000:
            print("{}: {}".format(x, count))

# print(df_im.head())
# print(df_co.head())
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

