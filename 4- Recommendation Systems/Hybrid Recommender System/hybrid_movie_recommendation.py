# User Based Recommendation

###############################################
# Görev 1 - Veriyi Hazırlama
###############################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

movie = pd.read_csv('Hafta4/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Hafta4/movie_lens_dataset/rating.csv')

df = movie.merge(rating, how='left', on='movieId')

"""

    movieId                    title                                       genres    userId  rating            timestamp
0         1         Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy   3.0000  4.0000  1999-12-11 13:36:47
1         1         Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy   6.0000  5.0000  1997-03-13 17:50:52
2         1         Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy   8.0000  4.0000  1996-06-05 13:37:51
        ...
"""

comment_counts = pd.DataFrame(df['title'].value_counts())

"""
En çok oy kullanılan filmler
                                     title
Pulp Fiction (1994)                        67310
Forrest Gump (1994)                        66172
Shawshank Redemption, The (1994)           63366
Silence of the Lambs, The (1991)           63299
Jurassic Park (1993)                       59715
                                            ...
"""

rare_movies = comment_counts[comment_counts['title'] < 1000].index
common_movies = df[~df['title'].isin(rare_movies)]
# Düşük oy kullanılma sayısına sahip filmleri çıkarıyoruz.

user_movie_df = common_movies.pivot_table(index=['userId'], columns=['title'], values='rating')
# 138493 rows x 3159 columns 'dan oluşan veri seti elde ettik. Buradaki 138493 sayısı eşsiz userId'lerini temsil
# etmektedir. 3159 ise filmleri temsil etmektedir.

def prepare_data():
    movie = pd.read_csv('Hafta4/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('Hafta4/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how='left', on='movieId')
    comment_counts = pd.DataFrame(df['title'].value_counts())
    rare_movies = comment_counts[comment_counts['title'] < 1000].index
    common_movies = df[~df['title'].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=['userId'], columns=['title'], values='rating')

    return user_movie_df

user_movie_df = prepare_data()

user_movie_df.head(20)

#%%


###############################################
# Görev 2 - Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
###############################################

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=7).values)
random_user_df = user_movie_df[user_movie_df.index == random_user]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
# Kullanıcının izlediği filmleri tespit ettik.


user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == 'Stargate (1994)']
# Kullanıcı 'Stargate (1994)' filmine 4 puan vermiştir.


#%%

###############################################
# Görev 3 - Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
###############################################

movies_watched_df = user_movie_df[movies_watched]
user_movie_count = movies_watched_df.T.notnull().sum().reset_index()
# Her userın öneri yapılacak kullanıcının izlediği filmlerin kaç tanesini izlediğini bulduk.

user_movie_count.columns = ['userId', 'movie_count']
users_same_movies = user_movie_count[user_movie_count['movie_count'] >= (len(movies_watched) * 60 / 100)][
                    'userId'].tolist()

len(users_same_movies)
# 8047 kullanıcı seçilen kullanıcının izlediği filmlerin yüzde 60 ve fazlasını izlemiştir.

#%%

###############################################
# Görev 4 - Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
###############################################

movies_watched_df_filtered = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

corr_df = movies_watched_df_filtered.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=['corr'])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df['user_id_1'] == random_user) & (corr_df['corr'] >= 0.65)][
    ['user_id_2', 'corr']].reset_index(drop=True)

top_users.rename(columns={'user_id_2': 'userId'}, inplace=True)

top_users_ratings = top_users.merge(rating[['userId', 'movieId', 'rating']], how='inner')

#%%

###############################################
# Görev 5 - Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
###############################################

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
recommendation_df = top_users_ratings.groupby('movieId').agg({'weighted_rating': 'mean'}).reset_index()

recommended_movies = recommendation_df[recommendation_df['weighted_rating'] > 3.5]. \
    sort_values('weighted_rating', ascending=False)

recommended_movies = recommended_movies.merge(movie[['movieId', 'title']])
recommended_movies['title'].head(5)
"""
0                    Net, The (1995)
1    Clear and Present Danger (1994)
2    Long Kiss Goodnight, The (1996)
3                Crimson Tide (1995)
4            Schindler's List (1993)
"""

# Item Based Recommendation

movie = pd.read_csv('Hafta4/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Hafta4/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how='left', on='movieId')

movieId = rating[(rating['userId'] == random_user) & (rating['rating'] == 5)].sort_values('timestamp', ascending=False)
current_movieId = movieId.iloc[0, 1]

movie_name = movie.loc[movie["movieId"] == current_movieId, 'title']
movie_name = user_movie_df[movie_name]

user_movie_df.corrwith(movie_name).sort_values(ascending=False)[1:6]
# Lord of the Rings: The Fellowship of the Ring, The (2001) filmi ile benzer beğenilme yapısına sahip olan filmleri
# bulduk