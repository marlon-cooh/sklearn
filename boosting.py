#!/usr/bin/env python3

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier #type:ignore
from sklearn.model_selection import train_test_split #type:ignore
from sklearn.metrics import accuracy_score #type:ignore

if __name__ == '__main__':
    df = pd.read_csv('./data/heart.csv')
    
    X = df.drop(['target'], axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    boost = GradientBoostingClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    boost_pred = boost.predict(X_test)
    print("-"*64)
    print(f"Accuracy score of Boosting {accuracy_score(boost_pred, y_test):.5f}")
    
    
    
    