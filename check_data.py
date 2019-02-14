import pandas as pd

df = pd.read_csv('movie-data/movie.metadata.tsv',delimiter='\t',encoding='utf-8')

#example
print(df.head())
#shape
print(df.shape)
#dimensions
print(df.ndim)
# columns names
print(df.columns)
# wikiid
print(df['wikiid'])
# title
print(df['title'])
# genres -> FIX
# print(df['genres'])

plot_df = pd.read_csv('movie-data/plot_summaries.txt', delimiter='\t')

print(plot_df.head(10))

word_list = [x.lower().replace(",","") for x in plot_df["summary"]]

print(len(word_list))
print(plot_df["wikiid"].tolist())

n_df = pd.DataFrame(columns=["title", "summary", "genres"])
for ind, x in enumerate(df['wikiid']):
    print(ind)
    if x in plot_df['wikiid'].tolist():
       n_df = n_df.append({"title":df["title"][ind], "summary":word_list[plot_df["wikiid"].tolist().index(x)]}, ignore_index=True)

print(n_df)



