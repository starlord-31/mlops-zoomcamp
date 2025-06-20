import pickle
import pandas as pd
import argparse
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True, help='Year of the taxi data (e.g., 2023)')
parser.add_argument('--month', type=int, required=True, help='Month of the taxi data (e.g., 3 for March)')
args = parser.parse_args()

# Load the model and DictVectorizer
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Define the categorical features
categorical = ['PULocationID', 'DOLocationID']

# Function to read and preprocess the data
def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

# Create data URL using CLI arguments
data_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{args.year}-{args.month:02d}.parquet'
df = read_data(data_url)

# Transform the data using DictVectorizer
dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

# Calculate standard deviation of the predicted durations
std_predicted_duration = np.std(y_pred)
print(f"Standard deviation of predicted duration: {std_predicted_duration:.2f}")

# Calculate and print mean predicted duration
mean_predicted_duration = np.mean(y_pred)
print(f"Mean predicted duration: {mean_predicted_duration:.2f}")

# Prepare the results DataFrame
df['ride_id'] = f'{args.year:04d}/{args.month:02d}_' + df.index.astype('str')
df_result = df[['ride_id']].copy()
df_result['predicted_duration'] = y_pred

# Save the results to a Parquet file
output_file = f'results_{args.year}_{args.month}.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

# Print the size of the output file
import os
file_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert to MB
print(f"Output file size: {file_size:.2f} MB")
