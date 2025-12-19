import pandas as pd

df = pd.read_csv("C:/workspace/genAi_&_ML/geeks_projects/machine_learning_project/data/raw/part2_tickets_multiclass_dirty.csv")

df.head()
df.info()
df.describe()
df.isna().sum()
