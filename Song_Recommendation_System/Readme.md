# Audio Content-Based Song Recommendation System with the GTZAN Dataset

The GTZAN Dataset (http://marsyas.info/downloads/datasets.html) was collected between 2000 and 2001, and used for the paper in genre classification " Musical genre classification of audio signals " by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002. Since then, it has been a popular choice for subsequent work on musical genre classification using machine learning.

This dataset has been mostly associated with music genre classification, a task that has lately been taken over entirely by deep learning models and convolutional neural networks. 
However, as specifying a genre for a song and its performer can be a very nebulous affair, given its subjective nature, even the most precise audio-based models often fail to reach a high accuracy. Especially when it comes to drawing boundaries between wide-reaching genres such as rock and pop, this precision becomes a matter of correct labeling as much as it is a matter of designing a good model.

Shifting away from genre classification, the goal of this small project will be to eventually create a recommendation model. In particular, it should be a recommendation based solely on the audio content of the song, uninfluenced by subjective associations not directly stemming from its sound. 

The instance of the dataset used in this work has been imported from Kaggle (https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) as curated by Andrada Olteanu. Credit should also go to her for providing well documented examples of how the features were extracted, and a basis for building ML models around these features.

The structure followed is:

*   An exploration of the features used to characterize each song in the database, what they correspond to at a high level, and how they might correlate to each other.
*   A Genre Classification model is built based on these extracted features, using a Cross Gradient Booster, which is one of the most powerful and accurate 'conventional' machine learning algorithms. The goal of this is to obtain the importance of such features when performing a classification task, so that only the most insightgul are used in the recommendation algorithm.
*   The recommendation algorithm is defined, using a few selected features and cosine similarity as a metric of how similar sounding songs are to each other. 

### Contents:

1. **Notebook_GTZAN.ipynb** is the Jupyter Notebook describing the method for analyzing the dataset, building a genre classification model, and eventually the recommender system. It is recommended as an introduction to the work here. If downloaded, it can be used to obtain the audio files and actually listen to the recommended songs.
2. **feature_pipeline.py** is the Python program used to extract features from the audio dataset and assemble a feature dataframe.
3. **feature_dataframe.csv** is the dataframe created by the above pipeline, used to train the classification model and build the recommendation system.
4. **XGB_genre_classification.py** is the Python program used to build and assess a Cross Gradient Boosting classifier model using the features contained in the dataframe.
5. **song_recommendation.py** is the program containing the SongRecommendation class, which is the recommender system built in this project. This system uses the features extracted from the audio files to estimate the most similar songs and provide recommendations.

