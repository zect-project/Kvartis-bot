import pandas as pd

df = pd.read_csv("Kvartirs/data/main_train.csv")

print(df["city"].value_counts())