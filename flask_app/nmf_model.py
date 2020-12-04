#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:34:42 2020

@author: hanbo
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from joblib import load

#load nmf model
nmf = load('my_nmf.joblib')

#global vars
COLUMN_NAMES = list(np.arange(1, 1683, dtype=int))
Q = nmf.components_
TITLES = pd.read_csv('titles.csv', index_col=('movie_id'))

#functions
def nmf_prediction(user_input: dict, nmf = nmf) -> list:
    """
    Takes user ratings of top 5 movies in 100K database and returns a 
    list of recommended titles based on nmf

    Parameters
    ----------
    user_input : dict
        A dictionary of ratings for the top five movies.

    Returns
    -------
    list
        A list of 5 recommended movies.

    """
    user_df = user_to_df(user_input)
    user_P = nmf.transform(user_df)
    #print(f'User_P has shape: {user_P.shape}')
    user_R = pd.DataFrame(np.dot(user_P, Q), index=[944], columns=COLUMN_NAMES)
    #print(f'User R shape: {user_R.shape}')
    #drop the five rated movies
    recos = user_R.drop(columns=list(user_input.keys()))
    #print(f'Shape of recos; {recos.shape}')
    recos = recos.sort_values(by=944, ascending=False, axis=1)
    #print(f'Shape of recos; {recos.shape}')
    top_five_indices= list(recos.iloc[0][:5].index)
    reco_titles = id_to_title(top_five_indices)
    #print(f'Reco_titles: {reco_titles}')
    return reco_titles



def id_to_title(movie_ids: list) -> list:
    '''
    Map movie id to movie title in TITLES dataframe

    Parameters
    ----------
    movie_ids : list
        movie ids.

    Returns
    -------
    list
        movie titles.

    '''
    rec_titles = []
    for i in movie_ids:
        rec_titles.append(TITLES.iloc[i][0])
    return rec_titles

def user_to_df(user_input: dict):
    """
    

    Parameters
    ----------
    user_input : dict
        DESCRIPTION.

    Returns
    -------
    DataFrame
        DESCRIPTION.

    """
    user_df = pd.DataFrame(user_input, index=[944], columns=COLUMN_NAMES)
    user_df = user_df.fillna(0)
    return user_df


if __name__ == "__main__":
    
    new_user_input1 = {50:4, 100:0, 181:0, 258:4, 174:0}
    new_user_input2 = {50:3, 100:2, 181:2, 258:2, 174:0}
    new_user_input3 = {50:0, 100:0, 181:0, 258:0, 174:0}
    #print('user input: ', new_user_input)
    test_list1 = nmf_prediction(new_user_input1, nmf)
    test_list2 = nmf_prediction(new_user_input2, nmf)
    test_list3 = nmf_prediction(new_user_input3, nmf)
    
