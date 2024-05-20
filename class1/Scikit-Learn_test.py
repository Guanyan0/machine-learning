import numpy as np

X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X, y)

X_test = np.array([[2, 3], [1, 2], [2, 0.5], [3, 2], [1, 2], [1, 1]])
y_test = np.array([1, 1, 0, 1, 1, 0])
y_pred = lr_model.predict(X_test)

print("Prediction on training set:", y_pred)
print("Accuracy on training set:", lr_model.score(X_test, y_test))
