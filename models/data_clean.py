import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load Data
original_data = pd.read_csv('spotify_songs.csv', sep=',' , header = 0)
data = original_data.copy()
image_data = pd.read_csv('Music.csv', sep=',', header=0)
data.drop(['playlist_name', 'playlist_id'], axis=1, inplace=True)

# Data Cleaning
scaler = MinMaxScaler()
numeric_features = ['loudness', 'instrumentalness']
data[numeric_features] = scaler.fit_transform(data[numeric_features])
data.columns = [col.replace('_', ' ') for col in data.columns]
data.rename(columns= {'track id': "Unique ID", 'track name': 'Name', 'track artist': 'Artist', 'track popularity'
                      : 'Popularity', 'track album id': 'Album ID', 'track album name': 'Album name',
                      'track album release date': 'Release date',
                      'playlist genre': 'Genre', 'playlist subgenre': 'Subgenre'}, inplace = True)
data.drop_duplicates(subset='Name', inplace=True)
data.dropna(inplace=True)

image_data = image_data[image_data['img'] != 'no']
image_data = image_data[['name', 'artist', 'img']]
data = image_data.merge(data, left_on=['name', 'artist'], right_on=['Name', 'Artist'], how='inner')


# Search Engine Data:
SE_data = data.copy()
SE_data = SE_data[['Name', 'Artist', 'Popularity', 'img', 'Release date', 'Genre', 'Subgenre']]
SE_data.to_csv('search_engine_data.csv', index=False)

# Recommendation Data:
recommendation_data = data.copy()
recommendation_data.to_csv('recommendation_data.csv', index=False)

# Popular Data:
popular_data = data.copy()
popular_data = popular_data[['name', 'artist', 'img', 'Popularity', 'Genre', 'Subgenre']]
popular_data.sort_values(by='Popularity', ascending=False, inplace=True)
popular_data.drop_duplicates(subset=['name'], inplace=True)
popular_data.to_csv('image_data.csv', index=False)

# Popular Artists:
popular_artists = data.copy()
popular_artists = popular_data.groupby('artist').agg({'Popularity': 'mean', 'img': 'first' }).reset_index()
popular_artists.sort_values(by='Popularity', ascending=False, inplace=True)
popular_artists.head(6).to_csv('popular_artists.csv', index=False)