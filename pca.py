#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #type:ignore
from sklearn.decomposition import IncrementalPCA #type:ignore
from sklearn.linear_model import LogisticRegression #type:ignore
from sklearn.preprocessing import StandardScaler #type:ignore
from sklearn.model_selection import train_test_split #type:ignore


if __name__ == "__main__":
    df_heart = pd.read_csv('./data/heart.csv')
    print(df_heart.head())
    
    # Features
    df_features = df_heart.drop(['target'], axis=1)
    df_target = df_heart['target']
    
    df_features = StandardScaler().fit_transform(df_features)
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)