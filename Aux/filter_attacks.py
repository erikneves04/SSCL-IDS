import pandas as pd

df = pd.read_csv("datasets/CICIDS-2017/dataset.csv")
print(df.columns)
print(df['Label'].unique())


filter = df['Label'] == 0
print('Attack packages count: ' + str(df[~filter].size))
print('Normal packages count: ' + str(df[filter].size))