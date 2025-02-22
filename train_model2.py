import mlflow
import mlflow.sklearn
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


mlflow.set_tracking_uri("https://8e23-89-75-182-190.ngrok-free.app")  # Lokalny MLflow
experiment_name = "iris-model"
mlflow.set_experiment(experiment_name)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.log_param("max_iter", 1000)


    model_path = "iris_model.pkl"
    joblib.dump(model, model_path)
    

    mlflow.sklearn.log_model(model, "model")

    print(f"Model Accuracy: {accuracy:.4f} (zapisany to MLflow)")

