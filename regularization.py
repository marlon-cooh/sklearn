#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split #type:ignore
from sklearn.linear_model import LinearRegression, Lasso, Ridge #type:ignore
from sklearn.metrics import mean_squared_error #type:ignore

if __name__ == "__main__":
    dataset = pd.read_csv('./data/whr2017.csv')
    X = dataset.drop(['score'], axis=1)
    Y = dataset['score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=False)
    
    model_linear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = model_linear.predict(X_test)
    linear_score = mean_squared_error(y_test, y_predict_linear)
    
    model_lasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = model_lasso.predict(X_test)
    lasso_score = mean_squared_error(y_test, y_predict_lasso)
    
    model_ridge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = model_ridge.predict(X_test)
    ridge_score = mean_squared_error(y_test, y_predict_ridge)
    
    print("-"*25)
    print(f"Linear Score {linear_score:.4f}")
    print("-"*25)
    print(f"Lasso Score {lasso_score:.4f}")
    print("-"*25)
    print(f"Ridge Score {ridge_score:.4f}")