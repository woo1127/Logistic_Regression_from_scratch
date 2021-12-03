from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
print(data.keys())

X, y = data['data'], data['target']

model = LogisticRegression()
model.fit(X, y)

print(model.predict())
print(model.accurate())
