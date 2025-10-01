import mlflow
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from joblib import dump
from joblib import load

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

data = shuffle(data, random_state=42)

batch_size = 60
batches = [data.iloc[i:i + batch_size] for i in range(0, len(data), batch_size)]

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Online Training Iris")
experiment_name = "Online Training Iris"

experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"])
latest_run_id = runs.iloc[0]["run_id"] 

artifact_uri = f"runs:/{latest_run_id}/model_artifacts/online_model.joblib"
local_path = mlflow.artifacts.download_artifacts(artifact_uri)

model = load(local_path)

with mlflow.start_run():
    mlflow.autolog() 
 
    X_batch = batches[0].drop(columns=['target'])
    y_batch = batches[0]['target']
       
    model.partial_fit(X_batch, y_batch)
 
    batch_accuracy = model.score(X_batch, y_batch)
    mlflow.log_metric("batch_accuracy", batch_accuracy)
    dump(model, "online_model.joblib")
 
    mlflow.log_artifact("online_model.joblib", artifact_path="model_artifacts")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="online_model",
        input_example=X_batch.iloc[:5]
    )

