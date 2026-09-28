import pandas as pd
from src.flaws.imbalance import inject_imbalance

df = pd.read_csv("data/raw/breast_cancer.csv")
print("original:", df["target"].value_counts().to_dict())

for share in [0.3, 0.1, 0.02]:
    flawed, info = inject_imbalance(df, minority_share=share, random_state=42)
    print("flawed:", flawed["target"].value_counts().to_dict(), info)