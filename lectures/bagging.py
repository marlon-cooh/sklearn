#!/usr/bin/env python3

import pandas as pd
from sklearn.ensemble import BaggingClassifier #type:ignore
from sklearn.neighbors import KNeighborsClassifier #type:ignore
from sklearn.model_selection import train_test_split #type:ignore
from sklearn.metrics import accuracy_score #type:ignore

if __name__ == '__main__':
    df = pd.read_csv('./data/heart.csv')
    
    X = df.drop(['target'], axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    print("-"*32)
    print(f"Accuracy score of knn = {accuracy_score(knn_pred, y_test):.5f}")
    
    bag_class = BaggingClassifier(
        estimator=KNeighborsClassifier(), # Assemblying by KNeighbors
        n_estimators=50, # Parallel runs
        random_state=42
    ).fit(X_train, y_train)
    
    bag_pred = bag_class.predict(X_test)
    print("-"*32)
    print(f"Accuracy score of KNN based Bagging classifier = {accuracy_score(bag_pred, y_test):.5f}")
    