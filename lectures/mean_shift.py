#!/usr/bin/env python3

import pandas as pd
from seaborn import scatterplot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.cluster import MeanShift #type:ignore

if __name__ == '__main__':
    candy_df = pd.read_csv('./data/candy.csv')
    X = candy_df.drop('competitorname', axis=1)
    
    meanshift = MeanShift().fit(X)
    print(meanshift.labels_) # Clusters created by mean shift model.
    print("Number of clusters given by Mean Shift: ", max(meanshift.labels_) + 1)
    print("-"*64)
    print(meanshift.cluster_centers_) # Coordinates of cluster centers.
    
    candy_df['meanshift'] = meanshift.labels_
    scatterplot(
        data=candy_df,
        x='sugarpercent',
        y='winpercent',
        hue='meanshift'
    )
    plt.tight_layout()
    plt.xlabel('Sugar (%)')
    plt.ylabel('Win (%)')
    plt.savefig('meanshift.png')
