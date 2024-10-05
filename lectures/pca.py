#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #type:ignore
from sklearn.decomposition import IncrementalPCA #type:ignore
from sklearn.linear_model import LogisticRegression #type:ignore
from sklearn.preprocessing import StandardScaler #type:ignore
from sklearn.model_selection import train_test_split #type:ignore
import matplotlib
matplotlib.use('Agg')  # Forces Matplotlib to run in non-GUI mode

if __name__ == "__main__":
    df_heart = pd.read_csv('./data/heart.csv')
    print(df_heart.shape)
    
    # Features
    df_features = df_heart.drop(['target'], axis=1)
    df_target = df_heart['target']
    
    df_features = StandardScaler().fit_transform(df_features)
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)
    
    # n_components = min(n_samples, n_features)
    pca = PCA(n_components=3)
    pca.fit(X_train)
    
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)
    
    x = range(len(pca.explained_variance_)) # Three components already defined
    y = pca.explained_variance_ratio_
    plt.plot(x, y)
    plt.xlabel('n_components')
    plt.ylabel('variance ratio')
    plt.savefig('./pca.png')
    
    logistic = LogisticRegression(solver='lbfgs', random_state=42)
    
    # Applying PCA using logistic as predicting model
    df_train = pca.transform(X_train)
    df_test = pca.transform(X_test)
    logistic.fit(df_train, y_train)
    print("Score PCA: ", logistic.score(df_test, y_test))
    
    df_train = ipca.transform(X_train)
    df_test = ipca.transform(X_test)
    logistic.fit(df_train, y_train)
    print("Score IPCA:", logistic.score(df_test, y_test))