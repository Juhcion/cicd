import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

X, y = load_iris(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = joblib.load("iris_model.pkl")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

threshold = 0.9

print(f"Accuracy: {accuracy:.4f}")

if accuracy >= threshold:
    print("Model spełnia wymagania dotyczące dokładności")
    sys.exit(0)  
else:
    print("Model nie spełnia wymagań dotyczących dokładności")
    sys.exit(1)  
