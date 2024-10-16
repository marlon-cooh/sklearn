#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV #type:ignore
from sklearn.ensemble import RandomForestRegressor #type:ignore

if __name__ == "__main__":
    
    # Import data
    df = pd.read_csv('../data/felicidad.csv')
    X = df.drop(['country', 'rank', 'score'], axis=1)
    y = df[['score']]
    
    # Params
    reg = RandomForestRegressor()
    params = {
        'n_estimators' : range(4, 16),
        'criterion' : ['absolute_error', 'squared_error'],
        'max_depth' : range(2, 11)
    }
    
    # Training model
    randcv_estim = RandomizedSearchCV(
        reg,
        params,
        n_iter=10,
        cv=3,
        scoring='neg_mean_absolute_error'
    ).fit(X, y)
    
    print(randcv_estim.best_estimator_)
    print(randcv_estim.best_params_)
    print(randcv_estim.predict(X.loc[[0]]))
    