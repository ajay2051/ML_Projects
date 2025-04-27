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

numeric_df = df.select_dtypes(include=['number'])

# Now compute correlation
correlation = numeric_df.corr()

print(correlation["isFraud"].sort_values(ascending=False))

df["type"] = df["type"].map({"CASH_OUT": 1, "PAYMENT": 2,
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
df["isFraud"] = df["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(df.head())

from sklearn.model_selection import train_test_split
x = np.array(df[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(df[["isFraud"]])

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# Suppose x, y are already defined properly
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

print("xtrain shape:", np.shape(xtrain))
print("ytrain shape:", np.shape(ytrain))

# Convert to DataFrames for better handling
xtrain_df = pd.DataFrame(xtrain)
xtest_df = pd.DataFrame(xtest)
ytrain_df = pd.DataFrame(ytrain)
ytest_df = pd.DataFrame(ytest)

# Examine data types of each column
print("Feature data types:", xtrain_df.dtypes)
print("Target data type:", ytrain_df.dtypes)

# Handle mixed data types in features - convert any object columns to numeric
for col in xtrain_df.columns:
    if xtrain_df[col].dtype == 'object':
        # If the column has string values, encode them
        le = LabelEncoder()
        # Combine train+test values for consistent encoding
        all_values = np.concatenate([xtrain_df[col].fillna('missing'), xtest_df[col].fillna('missing')])
        le.fit(all_values)
        xtrain_df[col] = le.transform(xtrain_df[col].fillna('missing'))
        xtest_df[col] = le.transform(xtest_df[col].fillna('missing'))
    else:
        # For numeric columns, just fill NaN values
        xtrain_df[col] = xtrain_df[col].fillna(0)
        xtest_df[col] = xtest_df[col].fillna(0)

# Handle target variable - ensure it's numeric
if ytrain_df.dtypes.iloc[0] == 'object':
    le = LabelEncoder()
    # Combine train+test values for consistent encoding
    all_values = np.concatenate([ytrain_df.iloc[:, 0].fillna('missing'), ytest_df.iloc[:, 0].fillna('missing')])
    le.fit(all_values)
    ytrain_clean = le.transform(ytrain_df.iloc[:, 0].fillna('missing'))
    ytest_clean = le.transform(ytest_df.iloc[:, 0].fillna('missing'))
else:
    # If already numeric, just fill NaN values
    ytrain_clean = ytrain_df.fillna(0).values.ravel()  # Flatten to 1D array
    ytest_clean = ytest_df.fillna(0).values.ravel()    # Flatten to 1D array

# Convert features back to numpy arrays
xtrain_clean = xtrain_df.values
xtest_clean = xtest_df.values

# Train the model with cleaned data
model = DecisionTreeClassifier()
model.fit(xtrain_clean, ytrain_clean)

# Test the model
print("Model accuracy:", model.score(xtest_clean, ytest_clean))

features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))