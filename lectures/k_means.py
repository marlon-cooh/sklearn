#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.cluster import KMeans #type:ignore

if __name__ == '__main__':
    candy_df = pd.read_csv('./data/candy.csv')
    X = candy_df.drop('competitorname', axis=1)
    
    kmeans = KMeans(
        n_clusters=4,
        init='k-means++',
        random_state=42
    ).fit(X)
    print(f"Centers: {len(kmeans.cluster_centers_)}")
    print("-"*64)
    print(kmeans.predict(X))
    
    candy_df['kmeans'] = kmeans.predict(X)
    sns.scatterplot(
        data=candy_df,
        x='sugarpercent',
        y='winpercent',
        hue='kmeans'
    )
    plt.tight_layout()
    plt.xlabel('Sugar (%)')
    plt.ylabel('Win (%)')
    plt.savefig('kmeans.png')
    
    