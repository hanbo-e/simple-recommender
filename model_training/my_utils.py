#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:48:35 2020

@author: hanbo
"""

def diagnose_df(df):
    print(f"df shape: {df.shape} \n")
    print(f"df data types: {df.dtypes} \n")
    print(df.head(5))