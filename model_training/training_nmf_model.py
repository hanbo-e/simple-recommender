#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:34:32 2020

@author: hanbo
"""

#Non-Negative Matrix Factorization of MovieLens 100K Dataset
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from joblib import dump, load

import my_utils


#load data
path = 'data/ml-100k/u.data'
df = pd.read_csv(path, sep='\t', header=(None))

col_names = {0:'user_id', 1:'item_id', 2:'rating', 3:'timestamps'}
df.rename(columns = col_names, inplace=True)
diagnose_df(df)

#transform to movies as cols, users as rows
df_trans = df.pivot(index='user_id', columns='item_id', values='rating')
diagnose_df(df_trans)

#option to use rating mean to fill NANs
#total_mean = df_trans.mean().mean()

#fill nans with 0
R = df_trans.fillna(0)
R.head()

#write R to disk
#R.to_csv('./data/R.csv')

#train nmf
nmf = NMF(n_components=20, max_iter=1000)
nmf.fit(R)
Q = nmf.components_
Q.shape #should have a shape of 20*1682

P = nmf.transform(R)
P.shape #

#save trained NMF model
dump(nmf, 'my_nmf.joblib')

#load model
#nmf = load('my_nmf.joblib')

error = round(nmf.reconstruction_err_, 2) 
error
R_hat = pd.DataFrame(np.dot(P, Q), columns=df_trans.columns, index=df_trans.index)
R_hat

#prediction
# Create a dictionary for a new user
new_user_input = {50:3, 100:2, 181:0, 258:4, 174:0}
new_user_input

# Convert it to a pd.DataFrame
new_user = pd.DataFrame(new_user_input, index=[944], columns=df_trans.columns)
new_user.shape
new_user.iloc[0][175]

new_user = new_user.fillna(0)

#Generate user_P 
user_P = nmf.transform(new_user)
user_P.shape

#New user R
user_R = pd.DataFrame(np.dot(user_P, Q), index=[944], columns=df_trans.columns)
user_R.shape

#remove films that are already seen, and return a zip of film title and rating, sorted by highest rating
recommendations = user_R.drop(columns=new_user_input.keys())
recommendations
recommendations.sort_values(by=944, ascending=False, axis=1)

#get top five movies for website and their title from 'u.item' file
movie_means = R.mean().sort_values(ascending=False)
top_five_index = movie_means.index[:5]
path3 = 'data/ml-100k/u.item'
df_titles = pd.read_csv(path3, sep='|', header = (None), encoding='latin-1')
col_names3 = {0:'movie_id', 1:'movie_title', 2:'release_date', 3:'video_release_date', 4: 'IMDB_url', 5: 'unknown', 6:'Action', 7:'Adventure', 8:'Animation', 9:'Childrens', 10:'Comedy', 11:'Crime', 12:'Documentary', 13:'Drama', 14:'Fantasy', 15:'Film_Noir', 16:'Horror', 17:'Musical', 18:'Mystery', 19:'Romance', 20:'Sci_Fi', 21:'Thriller', 22:'War', 23:'Western'}
df_titles.rename(columns = col_names3, inplace=True)
diagnose_df(df_titles)

#save a csv file of movie titles and movie id to use in recommender systems
titles  = df_titles[['movie_id', 'movie_title']]
titles.set_index('movie_id', inplace=True)
titles.to_csv('./data/titles.csv')

top_five_titles = []
for i in top_five_index:
    top_five_titles.append(df_titles.iloc[(i-1)][1])

movie_indices = []
for title in top_five_titles:
    my_index = df_titles.movie_title[df_titles['movie_title']== title].index.tolist()
    movie_indices.append(my_index)


    