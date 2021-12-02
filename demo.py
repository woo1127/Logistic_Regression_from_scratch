from logistic_regression import LogisticRegression
import pandas as pd

data = pd.read_csv('breast_cancer.csv')
data = data.drop(['id'], axis=1)

X = df.iloc[:, 1:]
print(X.columns)

y = df['diagnosis']
print(y)

model = LogisticRegression()
model.fit(X, y)

print(model.predict())
print(model.accurate())
