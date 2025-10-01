import pandas as pd
import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from joblib import dump

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_experiment("Online Training Iris")

model = SGDClassifier(loss="log_loss", learning_rate="adaptive", eta0=0.01, max_iter=10000)

classes = data["target"].unique()

with mlflow.start_run():
    mlflow.autolog()  

    X_batch = data.drop(columns=['target'])
    y_batch = data['target']
 
    model.partial_fit(X_batch, y_batch, classes=classes)
 
    accuracy = model.score(X_batch, y_batch)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy:.4f}")
    dump(model, "online_model.joblib")
 
    mlflow.log_artifact("online_model.joblib", artifact_path="model_artifacts")
 
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="online_model",
        input_example=X_batch.iloc[:5]
    )