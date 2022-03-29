from sklearn.datasets import load_breast_cancer
from logistic_regression import LogisticRegression
import pandas as pd

data = load_breast_cancer()
print(data.keys())

X, y = data['data'], data['target']

model = LogisticRegression()
model.fit(X, y)

print(model.predict())
print(model.accuracy())
