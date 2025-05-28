import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Taxi-Duration-Prediction")
mlflow.sklearn.autolog()

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    with mlflow.start_run():


        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        mlflow.log_artifact(local_path='/workspaces/mlops-zoomcamp/02-experiment_tracking/Homework/output/dv.pkl', artifact_path='preprocessor')

        mlflow.sklearn.log_model(rf, artifact_path="model")



if __name__ == '__main__':
    run_train()
