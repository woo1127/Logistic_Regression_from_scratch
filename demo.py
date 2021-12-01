from sklearn.datasets import load_breast_cancer
from logistic_regression import LogisticRegression

data = load_breast_cancer()

X, y = data['data'], data['target']

model = LogisticRegression()
model.fit(X, y)

print(model.predict())
print(model.accurate())
