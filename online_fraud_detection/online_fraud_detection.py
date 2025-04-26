import pandas as pd
import numpy as np

df = pd.read_csv("credit_card.csv")
df.head()

df.isnull().sum()

df.type.value_counts()

type = df["type"].value_counts()
transactions = type.index
quantity = type.values

import plotly.express as px
figure = px.pie(df,
             values=quantity,
             names=transactions,hole = 0.5,
             title="Distribution of Transaction Type")
figure.show()