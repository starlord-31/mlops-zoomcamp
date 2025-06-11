import prefect
import pandas as pd
import mlflow
import pickle
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

print("Prefect version:", prefect.__version__)

from prefect import flow, task

@task
def load_data():
    path = '/workspaces/mlops-zoomcamp/03-orchestration/yellow_tripdata_2023-03.parquet'
    df = pd.read_parquet(path)
    print(f"Records loaded: {len(df)}")
    return df

@task
def prepare_data(df):
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df['duration'].dt.total_seconds() / 60
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]
    
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    print(f"Records after preparation: {len(df)}")
    return df

@task
def train_model(df):
    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration'].values
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    print(f"Model intercept: {lr.intercept_}")
    return lr, dv

@task
def register_model(model, dv):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "linear_regression")
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("intercept", model.intercept_)
        
        with open("dict_vectorizer.pkl", "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact("dict_vectorizer.pkl", artifact_path="artifacts")
        print("Model and DictVectorizer registered in MLflow.")
        
        model_uri = mlflow.get_artifact_uri("model")
        model_path = model_uri.replace("file://", "")
        mlmodel_file_path = os.path.join(model_path, "model.pkl")
        
        if os.path.exists(mlmodel_file_path):
            mlmodel_size = os.path.getsize(mlmodel_file_path)
            print(f"MLmodel file size: {mlmodel_size} bytes")
        
        return mlmodel_size


@flow(name="nyc-taxi-pipeline")
def main_flow():
    df = load_data()
    df_clean = prepare_data(df)
    model, dv = train_model(df_clean)
    register_model(model, dv)


if __name__ == "__main__":
    main_flow()
    
