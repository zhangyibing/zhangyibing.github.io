import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('diabetes.csv')


print(df.head())

# the data has 768 rows and 9 columns
print(df.shape)

print(df.info())

# check 0 values in the data
zero_counts = (df==0).sum()
columns_with_zeros = zero_counts[zero_counts > 0]
print("Columns with counts of 0:")
print(columns_with_zeros)

# total_zeros_per_column = zero_counts[zero_counts > 0].sum()
# print("\nTotal count of zeros for each column:")
# print(total_zeros_per_column)

columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# replace 0s with NaN
df_copy = df.copy(deep=True)
df_copy[columns_with_zeros] = df_copy[columns_with_zeros].replace(0,np.NaN)

# print(df_copy)

df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# print(df_copy)

X = df.drop(columns='Outcome')
y = df['Outcome']

# print(X)
# print(y)

seed = 42
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

file_name = 'finalized_model.pkl'
pickle.dump(rf_model, open(file_name, 'wb'))




