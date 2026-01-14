#!/usr/bin/env python3
"""Compute average silhouette scores for selected classifications."""

import pandas as pd

# Read data
df_all = pd.read_csv('outputs/silhouette_binary_all.csv')
df_hair = pd.read_csv('outputs/silhouette_hair_color.csv')

# Selected classifications: bangs, heavy_makeup, hair_color, gender
selected = ['bangs', 'heavy_makeup', 'gender']

# Get data for each classification
dfs = []
for cls in selected:
    df_cls = df_all[df_all['Classification'] == cls][['Step', 'Baseline', 'CS_Mode', 'C_Mode']]
    dfs.append(df_cls.set_index('Step'))

# Add hair_color
df_hair_indexed = df_hair.set_index('Step')
dfs.append(df_hair_indexed)

# Compute average
df_avg = sum(dfs) / len(dfs)
df_avg = df_avg.reset_index()

print("Selected classifications: bangs, heavy_makeup, hair_color, gender")
print("\nAverage Silhouette Scores:")
print(df_avg.to_string(index=False))

# Save
df_avg.to_csv('outputs/silhouette_selected_average.csv', index=False)
print("\nSaved: outputs/silhouette_selected_average.csv")
