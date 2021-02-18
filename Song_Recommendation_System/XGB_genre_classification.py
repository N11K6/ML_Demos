#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function-based pipeline to train and assess a Cross Gradient Boosting
classifier model on the GTZAN dataset.

Requirements:
    Dependencies,
    Dataframe with GTZAN features and labels in .csv format

@author: NK
"""
#%% Dependencies:
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Warnings might pop up during cross validation, practical to ignore:
import warnings
warnings.filterwarnings('ignore', category=Warning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

#%%
def get_data(df_path):
    '''
    Get features and labels from dataframe.
    '''
    df = pd.read_csv(df_path)
    
    # Get features:
    X = df.loc[:, df.columns != 'genre'] 
    # Normalize features:
    cols = X.columns
    min_max_scaler = MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    
    X = pd.DataFrame(np_scaled, columns = cols)
    
    # Get labels
    y = df['genre']
    
    return X, y

#%%
def assess_model(model, X_train, X_test, y_train, y_test):
    '''
    Fits model on training data, makes predictions on test data,
    gives accuracy score.
    '''
    # Train model
    model.fit(X_train, y_train)
    # Make predictions
    preds = model.predict(X_test)
    # Print accuracy
    print('Accuracy :', round(accuracy_score(y_test, preds), 5), '\n') 

#%%
def print_results(results):
    '''
    Prints results of 5-fold cross validation.
    '''
    print('BEST PARAMS: {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
    
#%%
def XGB_CV(X_train, y_train, parameters):
    '''
    Performs 5-fold cross validation for XGB classifier using the training data
    and a specified dictionary of parameters to investigate
    '''
    # Ask permission to perform Cv:
    choice = input("Perform 5-fold cross validation (might take a while) [Y/N]?")
    
    if choice == 'Y':
        # Instantiate XGB model
        xgb = XGBClassifier(use_label_encoder=False, eval_metric = 'mlogloss')

        # 5-fold Cross Validation
        cv = GridSearchCV(xgb, parameters, cv=5)
        cv.fit(X_train, y_train.ravel())
        # Print results
        print_results(cv)
        # Return best parameters
        return cv.best_results_
    
    else:
        return 0  
#%%
def save_model(model, filename):
    '''
    Saves trained model.
    '''
    choice = input("Save this model [Y/N]?")
    if choice == 'Y':
        with open(filename,'wb') as f:
            pickle.dump(model,f)
    else:
        return 0
#%%
def main():
    '''
    Pipeline for model training (interactive)
    '''
    # Path to dataframe
    DF_PATH = 'feature_dataframe.csv'
    # Get features, labels
    X, y = get_data(DF_PATH)
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Default XGB parameters
    PARAMETERS = {
            'n_estimators': 800,
            'max_depth': 3,
            'learning_rate': 0.05
            }
    # Parameters to try in CV
    CV_PARAMETERS = {
            'n_estimators': [600, 800, 1000],
            'max_depth': [1, 3, 6],
            'learning_rate': [0.01, 0.05, 1]
            }
    # Filename to save trained model
    FILENAME = 'XGB_Music_Genre.pkl'
    # Get best parameters from CV
    BEST_PARAMETERS = XGB_CV(X_train, y_train, CV_PARAMETERS)
    
    # Choose what set of parameters to use
    choice = input(f'Default parameters are {PARAMETERS}, proceed [0] or use best parameters from cross validation [1]?')
    
    if choice == 1:
        XGB = XGBClassifier(parameters = BEST_PARAMETERS, eval_metric = 'mlogloss')
    else:
        XGB = XGBClassifier(parameters = PARAMETERS, eval_metric = 'mlogloss')
    
    # Assess model
    assess_model(XGB, X_train, X_test, y_train, y_test)
    # Save model
    save_model(XGB, FILENAME)
    
#%%
if __name__ == "__main__":
    main()