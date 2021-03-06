#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Song Recommender class using cosine similarity between features in dataframe to
estimate most similar songs and give them as recommendations.

Requirements:
    Dependencies,
    Feature dataframe in .csv format

Usage:
    Instantiate the class and call the similar_songs method specifying the
    title of the song as it appears in the dataframe, and the number of similar
    songs to output.

@author: NK
"""
# Dependencies:
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import cosine_similarity
#%%
class SongRecommender():
    '''
    Song Recommender class
    '''
    # Upon initialization:
    def __init__(self,
                 feature_dataframe
                 ):
        # Read from dataframe:
        self.dataframe = pd.read_csv(feature_dataframe, 
                                     index_col = 'filename'
                                     )
        # Calculate similarity matrix:
        self.similarity_matrix = self.construct_similarity_matrix()
        
    def construct_similarity_matrix(self):
        '''
        Constructs similarity matrix
        '''
        #Extract labels
        labels = self.dataframe[['genre']]
        
        # Drop labels from original dataframe
        df = self.dataframe.drop(columns=['genre', ])

        # Scale the data
        df_scaled = scale(df)
        
        # Cosine similarity
        similarity = cosine_similarity(df_scaled)

        # Convert into a dataframe and then set the row index and column names as labels
        sim_df_labels = pd.DataFrame(similarity)
        sim_df_names = sim_df_labels.set_index(labels.index)
        sim_df_names.columns = labels.index
        
        return sim_df_names
    
    def similar_songs(self,
                      song_name,
                      n_similar_songs
                      ):
        '''
        Finds similar songs based on cosine similarity between features
        '''
        # Find songs most similar to another song
        series = self.similarity_matrix[song_name].sort_values(ascending = False)
    
        # Remove cosine similarity == 1 (songs will always have the best match with themselves)
        series = series.drop(song_name)
    
        # Display the 5 top matches 
        print("\n*******\nSimilar songs to ", song_name)
        print(series.head(n_similar_songs))
        
        return series.index[:n_similar_songs]
