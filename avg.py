import pandas as pd

csv = pd.read_csv("result.csv", names=['inputs', 'targets', 'similarity'])
print(csv['similarity'].mean())

